package checkpoint

import (
	"fmt"
	"os"
	"sort"
)

// Entry 描述目录里一个正式检查点文件（.tmp 半截文件不算）。
type Entry struct {
	Path string
	Step int
	Era  uint64
}

// Entries 按 (step,era) 升序列出目录里全部正式检查点。
// 目录不存在时返回空列表（不是错误，与 Latest 的语义区分开：
// 调用方需要区分“什么都没有”时自行检查 len==0）。
func (s *Store) Entries() ([]Entry, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Entry
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		var stp int
		var era uint64
		if _, err := fmt.Sscanf(e.Name(), "ckpt-s%07d-e%07d.etc", &stp, &era); err == nil {
			out = append(out, Entry{Path: s.dir + string(os.PathSeparator) + e.Name(), Step: stp, Era: era})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Step != out[j].Step {
			return out[i].Step < out[j].Step
		}
		return out[i].Era < out[j].Era
	})
	return out, nil
}

// Prune 只保留最新的 keep 个正式检查点，删除其余，返回被删除的路径。
// 不变量：
//   - keep 会被钳制到 >=1：最新的一个检查点（下次启动恢复要用的点）绝不删除，
//     因此清理后重启绝不会看起来像全新作业；
//   - 只动正式文件，遗留 .tmp（半截写入的证据）保留，由 Load 路径按既有规则判定。
func (s *Store) Prune(keep int) (removed []string, err error) {
	if keep < 1 {
		keep = 1
	}
	entries, err := s.Entries()
	if err != nil {
		return nil, err
	}
	if len(entries) <= keep {
		return nil, nil
	}
	for _, e := range entries[:len(entries)-keep] {
		if rerr := os.Remove(e.Path); rerr != nil {
			return removed, fmt.Errorf("checkpoint: prune %s: %w", e.Path, rerr)
		}
		removed = append(removed, e.Path)
	}
	return removed, nil
}

// LoadUsable 从最新到最旧依次尝试加载正式检查点，返回第一个完整可用的，
// 以及被跳过（不可用）的检查点原因列表——调用方必须把这些原因如实记录/上报，
// 这是一次“明确采用仍然可用的点”，绝不是静默回退。
// 一个都不可用时的返回与既有语义一致：
//   - 目录里没有任何检查点 -> ErrMissing；
//   - 只有遗留 .tmp -> ErrPartialWrite；
//   - 有正式文件但全部不可用 -> 返回最新那个文件的判定错误（最相关的原因）。
func (s *Store) LoadUsable() (st *State, path string, skipped []error, err error) {
	entries, err := s.Entries()
	if err != nil {
		return nil, "", nil, err
	}
	if len(entries) == 0 {
		// 保持与 Load 一致的“空目录/只有 .tmp”区分。
		if _, lerr := s.Latest(); lerr != nil {
			return nil, "", nil, lerr
		}
		return nil, "", nil, ErrMissing
	}
	var firstErr error
	for i := len(entries) - 1; i >= 0; i-- {
		st, lerr := s.LoadPath(entries[i].Path)
		if lerr == nil {
			return st, entries[i].Path, skipped, nil
		}
		if firstErr == nil {
			firstErr = lerr
		}
		skipped = append(skipped, fmt.Errorf("%s: %w", entries[i].Path, lerr))
	}
	return nil, "", skipped, firstErr
}
