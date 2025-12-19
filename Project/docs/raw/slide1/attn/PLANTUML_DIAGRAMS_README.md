# PlantUML Diagrams for Attention Strategy Pattern Refactoring

This directory contains PlantUML diagrams documenting the Strategy Pattern refactoring of the attention backend system.

## Diagrams

### 1. Class Diagram (`attention_strategy_pattern.puml`)

**Purpose**: Shows the complete class structure of the refactored design.

**Key Elements**:
- `AttentionStrategy` - Abstract base class defining the strategy interface
- `FlashAttentionStrategy`, `NativeAttentionStrategy`, `XFormersAttentionStrategy` - Concrete implementations
- `AttentionStrategyFactory` - Factory with instance caching
- Module-level cached instances for zero-overhead performance
- Backend functions for backward compatibility
- Integration with existing `_AttentionBackendRegistry`

**View Online**: [PlantUML Online Editor](http://www.plantuml.com/plantuml/uml/)

### 2. Sequence Diagram (`attention_strategy_sequence.puml`)

**Purpose**: Illustrates the runtime execution flow when using the pattern.

**Flow**:
1. User calls backend function (e.g., `_flash_attention()`)
2. Function retrieves cached strategy instance
3. Strategy validates constraints
4. Strategy delegates to underlying library
5. Result returned to user

**Highlights**: 
- Shows zero instantiation overhead
- Demonstrates delegation pattern
- Illustrates backward compatibility

### 3. Architecture Overview (`attention_strategy_architecture.puml`)

**Purpose**: High-level component view of the entire system.

**Layers**:
- **Strategy Pattern Core**: Abstract base and concrete strategies
- **Performance Layer**: Module-level cached instances
- **Backward Compatibility Layer**: Existing function-based API
- **Existing Infrastructure**: Registry and dispatch system
- **External Libraries**: flash_attn, PyTorch, xFormers

**Highlights**:
- Shows separation between new design and legacy system
- Illustrates zero-overhead optimization
- Demonstrates no breaking changes

## How to View

### Option 1: Online PlantUML Viewer
1. Visit [PlantUML Online Editor](http://www.plantuml.com/plantuml/uml/)
2. Copy and paste the content of any `.puml` file
3. View the rendered diagram

### Option 2: VS Code Extension
1. Install the "PlantUML" extension in VS Code
2. Open any `.puml` file
3. Press `Alt+D` to preview

### Option 3: Command Line (requires PlantUML installed)
```bash
# Install PlantUML
# On macOS:
brew install plantuml

# On Ubuntu:
sudo apt-get install plantuml

# Generate PNG images
plantuml attention_strategy_pattern.puml
plantuml attention_strategy_sequence.puml
plantuml attention_strategy_architecture.puml
```

### Option 4: Using Docker
```bash
# Generate all diagrams
docker run --rm -v $(pwd):/data plantuml/plantuml *.puml

# This creates PNG files in the same directory
```

## Design Patterns Illustrated

### Strategy Pattern
- **Where**: `AttentionStrategy` and its concrete implementations
- **Benefit**: Each backend encapsulates its algorithm, making it easy to add new backends

### Factory Pattern
- **Where**: `AttentionStrategyFactory`
- **Benefit**: Centralizes strategy creation with caching for performance

### Singleton Pattern (variant)
- **Where**: Module-level cached instances
- **Benefit**: Zero instantiation overhead on hot paths

### Adapter Pattern
- **Where**: Backend functions wrapping strategies
- **Benefit**: Maintains backward compatibility with existing API

## Key Design Decisions

1. **Instance Caching**: Strategies are cached to avoid repeated instantiation
2. **Module-Level Cache**: Pre-instantiated strategies for hot paths
3. **Backward Compatibility**: Existing functions delegate to strategies
4. **Zero Overhead**: Performance benchmarks show no regression

## References

- Implementation: `Project/diffusers/src/diffusers/models/attention_dispatch.py`
- Documentation: `Project/docs/REFACTORING_STRATEGY_PATTERN.md`
- Summary: `Project/docs/IMPLEMENTATION_SUMMARY.md`
- Original Proposal: `Project/docs/raw/slide1/attn.md`
