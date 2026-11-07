package strategy

import (
	"fmt"
)

// Registry maps strategy names to Strategy instances.
type Registry struct {
	strategies map[string]Strategy
}

// NewRegistry creates a Registry with only the smart strategy.
func NewRegistry() *Registry {
	r := &Registry{strategies: make(map[string]Strategy)}

	// Only smart strategy is enabled - it's the default and only strategy.
	strategies := []Strategy{
		&StrategySmart{},
	}

	for _, s := range strategies {
		r.strategies[s.Name()] = s
	}
	return r
}

// Get retrieves a strategy by name.
func (r *Registry) Get(name string) (Strategy, error) {
	s, ok := r.strategies[name]
	if !ok {
		return nil, fmt.Errorf("unknown strategy: %q (available: %v)", name, r.Names())
	}
	return s, nil
}

// Names returns all registered strategy names.
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.strategies))
	for k := range r.strategies {
		names = append(names, k)
	}
	return names
}
