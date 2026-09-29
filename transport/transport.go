package transport

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"sync/atomic"
)

// ErrClosed 表示连接/服务已关闭。
var ErrClosed = errors.New("transport: closed")

// Envelope 是本地 IPC 的消息外壳，负载为原始 JSON，按 id 匹配请求与响应。
type Envelope struct {
	Type    string `json:"type"`
	Payload []byte `json:"payload,omitempty"`
	Error   string `json:"error,omitempty"`
	Code    string `json:"code,omitempty"`
	ID      int64  `json:"id"`
}

// Handler 处理一行请求并返回响应；返回 nil 时不回包。
type Handler func(ctx context.Context, req *Envelope) *Envelope

// Server 基于 unix domain socket 的本地 JSON-lines RPC 服务。
type Server struct {
	socketPath string
	listener   net.Listener
	handler    Handler

	mu      sync.Mutex
	closed  bool
	conns   map[net.Conn]struct{}
	wg      sync.WaitGroup
	onClose func(remoteAddr string)
}

func NewServer(socketPath string, handler Handler) *Server {
	return &Server{socketPath: socketPath, handler: handler, conns: map[net.Conn]struct{}{}}
}

// SetDisconnectHook 注册连接断开回调（用于心跳/死亡判定）。
func (s *Server) SetDisconnectHook(fn func(remoteAddr string)) { s.onClose = fn }

// Start 开始监听；若旧 socket 文件存在会先清理（本组件独占本地路径）。
func (s *Server) Start(ctx context.Context) error {
	_ = os.Remove(s.socketPath)
	l, err := net.Listen("unix", s.socketPath)
	if err != nil {
		return fmt.Errorf("transport: listen %s: %w", s.socketPath, err)
	}
	s.listener = l
	go func() {
		<-ctx.Done()
		s.stop()
	}()
	s.wg.Add(1)
	go s.acceptLoop(ctx)
	return nil
}

func (s *Server) acceptLoop(ctx context.Context) {
	defer s.wg.Done()
	for {
		c, err := s.listener.Accept()
		if err != nil {
			s.mu.Lock()
			closed := s.closed
			s.mu.Unlock()
			if closed {
				return
			}
			return
		}
		s.mu.Lock()
		s.conns[c] = struct{}{}
		s.mu.Unlock()
		s.wg.Add(1)
		go s.connLoop(ctx, c)
	}
}

func (s *Server) connLoop(ctx context.Context, c net.Conn) {
	defer func() {
		s.mu.Lock()
		delete(s.conns, c)
		s.mu.Unlock()
		c.Close()
		s.wg.Done()
		if s.onClose != nil {
			s.onClose(c.RemoteAddr().String())
		}
	}()
	// 每个连接一个 reader 和一个 writer，响应严格按请求行顺序返回，
	// 因此无需额外加锁；客户端也按相同顺序读取。
	reader := bufio.NewReader(c)
	writer := bufio.NewWriter(c)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, net.ErrClosed) {
				return
			}
			return
		}
		var req Envelope
		if jerr := json.Unmarshal(line, &req); jerr != nil {
			resp := &Envelope{ID: 0, Error: "bad json: " + jerr.Error(), Code: "bad_request"}
			b, _ := json.Marshal(resp)
			_, _ = writer.Write(append(b, '\n'))
			_ = writer.Flush()
			continue
		}
		resp := s.handler(ctx, &req)
		if resp == nil {
			resp = &Envelope{ID: req.ID}
		}
		resp.ID = req.ID
		b, merr := json.Marshal(resp)
		if merr != nil {
			continue
		}
		if _, werr := writer.Write(append(b, '\n')); werr != nil {
			return
		}
		if ferr := writer.Flush(); ferr != nil {
			return
		}
	}
}

func (s *Server) stop() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	l := s.listener
	for c := range s.conns {
		_ = c.Close()
	}
	s.mu.Unlock()
	if l != nil {
		_ = l.Close()
	}
	_ = os.Remove(s.socketPath)
	s.wg.Wait()
}

// Addr 返回监听的 socket 路径。
func (s *Server) Addr() string { return s.socketPath }

// Wait 阻塞直到服务完全停止。
func (s *Server) Wait() { s.wg.Wait() }

// Client 是与 Server 配对的本地客户端，断线后会自动重连。
type Client struct {
	socketPath string

	mu     sync.Mutex
	conn   net.Conn
	w      *bufio.Writer
	r      *bufio.Reader
	closed bool
	nextID int64
}

func NewClient(socketPath string) *Client { return &Client{socketPath: socketPath} }

func (c *Client) ensureConn() error {
	if c.conn != nil {
		return nil
	}
	conn, err := net.Dial("unix", c.socketPath)
	if err != nil {
		return fmt.Errorf("transport: dial %s: %w", c.socketPath, err)
	}
	c.conn = conn
	c.w = bufio.NewWriter(conn)
	c.r = bufio.NewReader(conn)
	return nil
}

// Call 发送一条请求并读取对应响应；连接损坏时自动重连并重试一次。
func (c *Client) Call(ctx context.Context, req *Envelope) (*Envelope, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, ErrClosed
	}
	if req.ID == 0 {
		req.ID = atomic.AddInt64(&c.nextID, 1)
	}
	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		if err := c.ensureConn(); err != nil {
			lastErr = err
			// 等待并重试（主要给服务刚启动留时间）。
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
			if attempt == 0 {
				continue
			}
			return nil, lastErr
		}
		b, _ := json.Marshal(req)
		if _, err := c.w.Write(append(b, '\n')); err != nil {
			c.teardownConn()
			lastErr = err
			continue
		}
		if err := c.w.Flush(); err != nil {
			c.teardownConn()
			lastErr = err
			continue
		}
		line, err := c.r.ReadBytes('\n')
		if err != nil {
			c.teardownConn()
			lastErr = err
			continue
		}
		var resp Envelope
		if err := json.Unmarshal(line, &resp); err != nil {
			lastErr = err
			continue
		}
		return &resp, nil
	}
	return nil, fmt.Errorf("transport: call failed: %w", lastErr)
}

func (c *Client) teardownConn() {
	if c.conn != nil {
		_ = c.conn.Close()
	}
	c.conn = nil
	c.w = nil
	c.r = nil
}

func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	if c.conn != nil {
		err := c.conn.Close()
		c.conn = nil
		return err
	}
	return nil
}
