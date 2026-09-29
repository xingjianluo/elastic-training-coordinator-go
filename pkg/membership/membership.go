package membership

import (
	"sort"
	"time"

	"github.com/xingjianluo/elastic-training-coordinator-go/pkg/sharding"
)

type State string

const (
	StateAlive     State = "alive"
	StateSlow      State = "slow"
	StateDead      State = "dead"
	StateCompleted State = "completed"
)

type Member struct {
	ID            string    `json:"id"`
	Generation    int       `json:"generation"`
	State         State     `json:"state"`
	LastHeartbeat time.Time `json:"-"`
}

type View struct {
	Generation int           `json:"generation"`
	Dimension  int           `json:"dimension"`
	Members    []Member      `json:"members"`
	Plan       sharding.Plan `json:"plan"`
}

// NewView builds a deterministic member order and matching shard plan. All
// members in a fresh generation start alive; replay state is applied by the
// coordinator before this view is exposed.
func NewView(generation, dimension int, now time.Time, ids []string) (View, error) {
	plan, err := sharding.PlanFor(dimension, ids)
	if err != nil {
		return View{}, err
	}
	members := make([]Member, len(plan.Workers))
	for i, id := range plan.Workers {
		members[i] = Member{ID: id, Generation: generation, State: StateAlive, LastHeartbeat: now}
	}
	return View{Generation: generation, Dimension: dimension, Members: members, Plan: plan}, nil
}

func SortedIDs(members []Member) []string {
	ids := make([]string, len(members))
	for i := range members {
		ids[i] = members[i].ID
	}
	sort.Strings(ids)
	return ids
}

// Classify distinguishes a node that has recently made progress but exceeded a
// soft deadline from one whose transport has been silent beyond the hard
// heartbeat deadline.
func Classify(m Member, now time.Time, softDeadline, hardDeadline time.Duration) State {
	if m.State == StateDead || m.State == StateCompleted {
		return m.State
	}
	silence := now.Sub(m.LastHeartbeat)
	switch {
	case silence >= hardDeadline:
		return StateDead
	case silence >= softDeadline:
		return StateSlow
	default:
		return StateAlive
	}
}

func (v View) Member(id string) (Member, bool) {
	for _, m := range v.Members {
		if m.ID == id {
			return m, true
		}
	}
	return Member{}, false
}

func (v View) IDs() []string { return SortedIDs(v.Members) }
