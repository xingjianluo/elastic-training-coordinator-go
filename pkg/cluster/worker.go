package cluster

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"
)

// GradientFn computes the gradient for one parameter shard from the
// authoritative shard parameters returned by the coordinator.
type GradientFn func(workerID string, gen, step int, shard sharding.Range, params []float64) []float64

type Worker struct {
	id              string
	heartbeatPeriod time.Duration
	dialTimeout     time.Duration
	gradient        GradientFn
	logf            func(string, ...any)
}

type WorkerOption func(*Worker)

func WithHeartbeat(d time.Duration) WorkerOption {
	return func(w *Worker) { w.heartbeatPeriod = d }
}

func WithDialTimeout(d time.Duration) WorkerOption {
	return func(w *Worker) { w.dialTimeout = d }
}

func NewWorker(id string, gradient GradientFn, opts ...WorkerOption) *Worker {
	w := &Worker{
		id:              id,
		heartbeatPeriod: 20 * time.Millisecond,
		dialTimeout:     2 * time.Second,
		gradient:        gradient,
		logf: func(format string, args ...any) {
			fmt.Fprintf(os.Stderr, format+"\n", args...)
		},
	}
	for _, opt := range opts {
		opt(w)
	}
	return w
}

// Run talks to one coordinator until it reports completion. It is safe to
// invoke again with the same worker ID after a process crash: the coordinator
// rejects old generation/step contributions and returns the current view.
func (w *Worker) Run(socketPath string) error {
	conn, err := Dial(socketPath)
	if err != nil {
		return fmt.Errorf("worker %s: connect: %w", w.id, err)
	}
	defer conn.Close()
	return w.runOnConn(conn)
}

func (w *Worker) runOnConn(conn net.Conn) error {
	writeCh := make(chan *Request, 32)
	writeDone := make(chan error, 1)
	go func() {
		writer := bufio.NewWriter(conn)
		enc := json.NewEncoder(writer)
		for req := range writeCh {
			if err := enc.Encode(req); err != nil {
				writeDone <- err
				return
			}
			if err := writer.Flush(); err != nil {
				writeDone <- err
				return
			}
		}
		writeDone <- nil
	}()

	var mu sync.Mutex
	seq := uint64(0)
	pending := map[uint64]chan Response{}
	readErr := make(chan error, 1)
	go func() {
		dec := json.NewDecoder(conn)
		for {
			var resp Response
			if err := dec.Decode(&resp); err != nil {
				readErr <- err
				return
			}
			mu.Lock()
			ch := pending[resp.ReqID]
			delete(pending, resp.ReqID)
			mu.Unlock()
			if ch != nil {
				ch <- resp
			}
		}
	}()

	call := func(reqType string, gen, step int, values []float64) (Response, error) {
		mu.Lock()
		seq++
		id := seq
		ch := make(chan Response, 1)
		pending[id] = ch
		mu.Unlock()
		req := &Request{Seq: id, WorkerID: w.id, Type: reqType, Generation: gen, Step: step, Values: values}
		select {
		case writeCh <- req:
		case err := <-writeDone:
			return Response{}, fmt.Errorf("worker %s: write stopped: %w", w.id, err)
		case err := <-readErr:
			return Response{}, fmt.Errorf("worker %s: connection closed: %w", w.id, err)
		}
		select {
		case resp := <-ch:
			return resp, nil
		case err := <-readErr:
			return Response{}, fmt.Errorf("worker %s: response closed: %w", w.id, err)
		case <-time.After(w.dialTimeout):
			return Response{}, fmt.Errorf("worker %s: response timeout", w.id)
		}
	}
	defer close(writeCh)

	view, err := call("join", 0, 0, nil)
	if err != nil {
		return err
	}
	if view.Code != CodeOK {
		if view.Code == CodeAlreadyCompleted {
			return nil
		}
		return fmt.Errorf("worker %s: join rejected: %s", w.id, view.Message)
	}

	stopHeartbeat := make(chan struct{})
	heartbeatDone := make(chan struct{})
	heartbeatGen := view.Generation
	go func() {
		defer close(heartbeatDone)
		ticker := time.NewTicker(w.heartbeatPeriod)
		defer ticker.Stop()
		gen := heartbeatGen
		for {
			select {
			case <-stopHeartbeat:
				return
			case <-ticker.C:
				resp, err := call("heartbeat", gen, 0, nil)
				if err == nil && resp.Code == CodeGenerationRestart {
					gen = resp.Generation
					heartbeatGen = resp.Generation
				}
			}
		}
	}()
	defer func() {
		close(stopHeartbeat)
		<-heartbeatDone
	}()

	for {
		if view.RunDone {
			return nil
		}
		owned, err := view.Plan.ShardFor(w.id)
		if err != nil {
			return fmt.Errorf("worker %s: current plan has no owned shard: %w", w.id, err)
		}
		params, err := shardParameters(view, w.id)
		if err != nil {
			return err
		}
		grad := w.gradient(w.id, view.Generation, view.Step, owned, params)
		req := &Request{Seq: 0, WorkerID: w.id, Type: "contribute", Generation: view.Generation, Step: view.Step, Shard: owned, Values: grad}
		mu.Lock()
		seq++
		req.Seq = seq
		ch := make(chan Response, 1)
		pending[req.Seq] = ch
		mu.Unlock()
		select {
		case writeCh <- req:
		case err := <-writeDone:
			return fmt.Errorf("worker %s: write stopped: %w", w.id, err)
		case err := <-readErr:
			return fmt.Errorf("worker %s: connection closed: %w", w.id, err)
		}
		var next Response
		select {
		case next = <-ch:
		case err := <-readErr:
			return fmt.Errorf("worker %s: response closed: %w", w.id, err)
		case <-time.After(w.dialTimeout):
			return fmt.Errorf("worker %s: response timeout", w.id)
		}
		if err != nil {
			return err
		}
		switch next.Code {
		case CodeOK, CodeAlreadyCompleted:
			view = next
			if view.RunDone || next.Code == CodeAlreadyCompleted {
				return nil
			}
		case CodeGenerationRestart:
			w.logf("worker %s decision: generation=%d step=%d rejected-reason=%s retry-generation=%d retry-step=%d",
				w.id, view.Generation, view.Step, next.Code, next.Generation, next.Step)
			view = next
		case CodeSlowWorker:
			w.logf("worker %s decision: slow-not-dead pending response=%s", w.id, next.Message)
			return fmt.Errorf("%w: %s", ErrSlowWorker, next.Message)
		case CodeWorkerDead:
			return fmt.Errorf("%w: %s", ErrWorkerDead, next.Message)
		default:
			return fmt.Errorf("%w: %s", ErrInvalidRequest, next.Message)
		}
	}
}

func shardParameters(view Response, workerID string) ([]float64, error) {
	owned, err := view.Plan.ShardFor(workerID)
	if err != nil {
		return nil, err
	}
	switch len(view.Values) {
	case view.Plan.Dimension:
		return append([]float64(nil), view.Values[owned.Start:owned.End]...), nil
	case owned.Len():
		return append([]float64(nil), view.Values...), nil
	default:
		return nil, fmt.Errorf("worker %s: received %d parameters for shard %v in dimension %d",
			workerID, len(view.Values), owned, view.Plan.Dimension)
	}
}

// Dial opens only a local Unix-domain socket; no TCP/network endpoint is used.
func Dial(socketPath string) (net.Conn, error) {
	return net.DialTimeout("unix", socketPath, 2*time.Second)
}

func IsClosed(err error) bool {
	return errors.Is(err, io.EOF) || errors.Is(err, net.ErrClosed)
}
