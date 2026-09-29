package checkpoint

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
)

// buildRaw 按磁盘格式拼出原始字节（测试用，可故意改坏）。
func buildRaw(st *State) []byte {
	body, _ := json.Marshal(st)
	sum := sha256.Sum256(body)
	hdr := make([]byte, headerLen)
	copy(hdr[0:8], magicText)
	binary.BigEndian.PutUint32(hdr[8:12], st.Version)
	binary.BigEndian.PutUint32(hdr[12:16], uint32(len(body)))
	return append(append(hdr, body...), sum[:]...)
}
