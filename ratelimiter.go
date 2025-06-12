package main

import (
	"sync"
	"time"
)

type RateLimiter struct {
	users map[int64]time.Time
	mu    sync.RWMutex
}

func NewRateLimiter() *RateLimiter {
	return &RateLimiter{
		users: make(map[int64]time.Time),
	}
}

func (rl *RateLimiter) Allow(userID int64) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	lastReq, exists := rl.users[userID]
	if !exists || time.Since(lastReq) > 10*time.Second {
		rl.users[userID] = time.Now()
		return true
	}
	return false
}
