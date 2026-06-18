# Refactoring Plan

This document defines the refactoring and cleanup plan for the first-party Mullama codebase.

It is intended to make the codebase:

- More modular
- Easier to extend safely
- Easier to test in isolation
- Easier to reuse across CLI, daemon, UI, and bindings
- Less dependent on large multi-responsibility files

## Scope

This plan applies to first-party code only:

- `src/`
- `src/daemon/`
- `src/bin/`
- `bindings/`
- `ui/src/`

This plan does not target vendored, generated, or third-party code:

- `llama.cpp/`
- `target/`
- `bindings/python/venv/`
- `bindings/node/node_modules/`
- lockfiles and generated binaries

## Refactoring Goals

Every refactor should improve at least one of these:

1. Single responsibility per file/module
2. Clear separation between transport, orchestration, domain logic, and FFI
3. Reusable core services shared across CLI, daemon, UI, and bindings
4. Reduced duplication between codepaths that currently implement similar behavior
5. Smaller public module surfaces with more intentional exports
6. Better unit-test seams for parsing, prompt building, model resolution, and streaming

## Core Rules

### 1. Separate layers

Code should be organized into distinct layers:

- Domain types and pure logic
- Service/orchestration logic
- Transport adapters
- Persistence/cache helpers
- UI/view logic
- FFI wrappers

Transport code must not own core business logic.

Examples:

- HTTP handlers should call daemon services, not assemble model resolution and loading logic inline.
- CLI commands should call reusable service functions, not reimplement daemon or Hugging Face workflows.
- Vue views should delegate state and side effects to composables/services.

### 2. Avoid file-level monoliths

Any file that owns multiple subsystems should be split.

Heuristics for splitting:

- More than one transport surface in the same file
- Request DTOs mixed with handler logic and middleware
- Domain types mixed with parsing and filesystem code
- Runtime orchestration mixed with low-level generation internals
- FFI wrappers mixed with higher-level placeholder APIs

### 3. Prefer vertical modules over flat mega-files

Prefer:

```text
src/daemon/openai/
  mod.rs
  router.rs
  middleware.rs
  types.rs
  chat.rs
  completions.rs
  models.rs
  defaults.rs
  system.rs
  ui.rs
```

Over:

```text
src/daemon/openai.rs
```

### 4. Reuse before reimplementing

If the same workflow exists in multiple places, consolidate it into a reusable module.

Known duplication targets:

- Hugging Face download and cache logic
- Model resolution logic
- Prompt/chat template fallback logic
- Streaming chunk formatting
- Settings/API key persistence in the UI

### 5. Keep public exports intentional

`mod.rs` files should not become giant re-export funnels unless that surface is deliberate.

Public exports should favor:

- Stable domain types
- Stable service entry points

Avoid exporting entire internal subsystems by default.

## High-Priority Refactor Targets

### 1. `src/daemon/openai.rs`

Current issues:

- Router setup, middleware, DTOs, handlers, metrics, system API, model API, defaults API, and embedded UI handling all live in one file
- OpenAI transport concerns are mixed with internal daemon management concerns
- Hard to test request mapping separately from handler execution

Target structure:

```text
src/daemon/openai/
  mod.rs
  router.rs
  middleware.rs
  error.rs
  types.rs
  chat.rs
  completions.rs
  embeddings.rs
  models.rs
  defaults.rs
  system.rs
  metrics.rs
  ui.rs
```

Rules:

- `router.rs` only wires routes and layers
- `middleware.rs` owns auth and rate limiting
- `types.rs` owns OpenAI HTTP request/response DTOs
- handler files convert HTTP payloads to daemon service calls
- model-management APIs should use reusable service helpers, not inline resolution logic

### 2. `src/daemon/server.rs`

Current issues:

- Daemon runtime, request dispatch, prompt construction, text generation, streaming, vision handling, embeddings, memory helpers, and builder live in one file
- Several private helpers are reusable but trapped in a monolith
- Text and vision streaming contain parallel logic that should share internals

Target structure:

```text
src/daemon/server/
  mod.rs
  daemon.rs
  config.rs
  builder.rs
  dispatch.rs
  prompt.rs
  embeddings.rs
  generation/
    mod.rs
    common.rs
    text.rs
    streaming.rs
    vision.rs
```

Rules:

- `dispatch.rs` maps `Request` to service methods
- `prompt.rs` owns prompt assembly and fallback chat formatting
- `generation/common.rs` owns stop-sequence handling and shared generation helpers
- `generation/text.rs` and `generation/vision.rs` own mode-specific execution
- builder logic moves out of the runtime implementation file

### 3. `src/multimodal.rs`

Current issues:

- Two conceptually different APIs coexist in one file
- A high-level multimodal abstraction with placeholder behavior is mixed with the actual mtmd-backed implementation used by the daemon
- Audio utilities, image types, and mtmd FFI wrappers are all flattened together

Target structure:

```text
src/multimodal/
  mod.rs
  types.rs
  config.rs
  audio.rs
  utils.rs
  high_level.rs
  mtmd/
    mod.rs
    bitmap.rs
    chunks.rs
    context.rs
```

Rules:

- The mtmd implementation is the real backend and should be isolated cleanly
- The high-level API should either be implemented fully or clearly marked experimental
- Placeholder logic should not sit beside production FFI wrappers without separation

### 4. Hugging Face integration

Current issues:

- Similar Hugging Face responsibilities are split across `src/huggingface.rs` and `src/daemon/hf.rs`
- Search, download, cache, and spec parsing are not clearly centralized
- CLI and daemon reuse is weaker than it should be

Target structure:

```text
src/hf/
  mod.rs
  types.rs
  spec.rs
  search.rs
  download.rs
  cache.rs
  resolve.rs
```

Rules:

- Keep one shared implementation for Hugging Face behavior
- CLI and daemon should consume the same core services
- Avoid separate downloader abstractions unless they solve different problems

### 5. `src/bin/mullama.rs`

Current issues:

- Argument definitions and command execution logic for many unrelated workflows live in one binary file
- CLI behavior is difficult to navigate and reuse
- Serve, cache, model, daemon, and Modelfile workflows are intermixed

Target structure:

```text
src/bin/mullama/
  main.rs
  args.rs
  commands/
    mod.rs
    serve.rs
    run.rs
    models.rs
    cache.rs
    pull.rs
    modelfile.rs
    daemon.rs
    status.rs
```

Rules:

- `args.rs` defines clap structures only
- command files own orchestration for a specific command family
- shared output formatting should live in a helper module

### 6. `src/daemon/models.rs`

Current issues:

- Config DTOs, stats, loaded-model runtime state, pool behavior, and manager logic live together
- Context pooling and model metadata are reusable concepts but not isolated

Target structure:

```text
src/daemon/models/
  mod.rs
  config.rs
  stats.rs
  loaded.rs
  pool.rs
  manager.rs
```

Rules:

- `config.rs` owns `ModelConfig` and `ModelLoadConfig`
- `stats.rs` owns counters and memory estimation
- `loaded.rs` owns `LoadedModel`
- `pool.rs` owns context-pool mechanics
- `manager.rs` owns orchestration and registry behavior

## Medium-Priority Refactor Targets

### `src/modelfile.rs`

Current issues:

- Domain types, parser, serializer, digest verification, and execution audit record are in one file

Target structure:

```text
src/modelfile/
  mod.rs
  types.rs
  parser.rs
  serialize.rs
  audit.rs
  fs.rs
```

### `src/builder.rs`

Current issues:

- Model, context, sampler, penalty, and presets are bundled together

Target structure:

```text
src/builder/
  mod.rs
  model.rs
  context.rs
  sampler.rs
  penalty.rs
  presets.rs
```

### `src/daemon/tui.rs`

Current issues:

- State, input handling, command execution, and rendering are coupled tightly

Target structure:

```text
src/daemon/tui/
  mod.rs
  state.rs
  input.rs
  commands.rs
  render.rs
  layout.rs
```

## Lower-Priority Or Cohesive Modules

These are large but mostly cohesive and should be split only after the higher-priority monoliths:

- `src/model.rs`
- `src/context.rs`
- `src/sampling.rs`
- `src/grammar.rs`
- `src/daemon/registry.rs`

The goal there is not arbitrary fragmentation. Only split them if:

- a clean sub-domain boundary emerges
- tests become easier
- duplication is removed
- public APIs become clearer

## UI Refactoring Plan

### `ui/src/api/client.ts`

Current issues:

- API key storage, auth headers, fetch wrapper, SSE parsing, DTOs, OpenAI API, and management API are all in one file

Target structure:

```text
ui/src/api/
  http.ts
  auth.ts
  sse.ts
  openai.ts
  management.ts
  types.ts
```

### `ui/src/views/*.vue`

Current issues:

- Views hold both page composition and feature state/effects
- Reusable pieces are still embedded in full pages

Target structure:

```text
ui/src/features/chat/
  composables.ts
  ConversationSidebar.vue
  ChatHeader.vue
  ChatComposer.vue
  MessageList.vue

ui/src/features/models/
  composables.ts
  DownloadedModels.vue
  ActiveModels.vue
  DefaultModelGrid.vue
  PullModelModal.vue

ui/src/features/settings/
  composables.ts
  SettingsForm.vue
  ThemeSettings.vue
  ModelDefaultsSettings.vue
```

Rules:

- Views compose features
- Composables own state, IO, and persistence
- Components own rendering and local interactions

## Bindings Refactoring Plan

### Python and Node bindings

Current issues:

- Large single binding files mirror multiple classes and services in one place

Target structure:

```text
bindings/python/src/
  lib.rs
  model.rs
  context.rs
  sampler.rs
  embedding.rs
  system.rs

bindings/node/src/
  lib.rs
  model.rs
  context.rs
  sampler.rs
  embedding.rs
  system.rs
```

Rules:

- Keep binding entrypoints thin
- Mirror the core domain modules where practical
- Shared conversion helpers should be centralized

## Reusability Standards

Any new module introduced by refactoring should follow these standards:

### Pure logic first

If logic can be written without IO, it should be.

Examples:

- model spec parsing
- stop-sequence merging
- prompt assembly
- metric formatting
- cache key generation

### Transport adapters second

HTTP, IPC, CLI, and UI code should adapt requests to shared services instead of re-owning logic.

### DTOs separate from domain models

Request/response wire formats should not become the domain model by default.

### Shared resolution services

The following should exist as reusable services rather than repeated logic:

- model name resolution
- Hugging Face spec parsing
- Ollama model config extraction
- Modelfile to runtime config conversion
- prompt rendering
- streaming chunk translation

## Cleanup Standards

Refactoring should also clean up the following:

### Delete or isolate placeholder code

If a subsystem is not production-ready, it should be either:

- completed
- clearly marked experimental
- moved behind a dedicated module boundary
- removed if unused

### Reduce duplicate helpers

Do not keep duplicate versions of:

- size formatting
- auth/header parsing
- error mapping
- stream parsing
- model spec resolution

### Tighten module exports

Prefer:

- `pub(crate)` internally
- explicit `pub use` lists
- narrow public API surfaces

Over:

- broad module-wide exposure
- flat re-export sprawl

## Execution Order

Recommended order of work:

1. Split `src/daemon/openai.rs`
2. Split `src/daemon/server.rs`
3. Split `src/multimodal.rs`
4. Consolidate Hugging Face code into shared modules
5. Split `src/daemon/models.rs`
6. Split `src/bin/mullama.rs`
7. Split `src/modelfile.rs`
8. Split `src/builder.rs`
9. Refactor TUI modules
10. Refactor UI API/client and feature views
11. Refactor bindings

## Acceptance Criteria

A refactor should be considered complete only if:

1. Existing behavior is preserved
2. Tests still pass, or equivalent coverage is added where behavior moves
3. No duplicate logic remains in old and new locations
4. New module boundaries are documented by names and folder structure
5. Transport layers are thinner than before
6. New code is easier to unit-test in isolation
7. Public exports are narrower or clearer than before

## What Not To Do

- Do not split files only to create shallow wrappers
- Do not move code without clarifying ownership boundaries
- Do not create circular module dependencies
- Do not keep both old and new implementations indefinitely
- Do not let handlers, CLI commands, and UI pages own core business logic

## Immediate Next Refactor

The first concrete refactor should be:

- Extract `src/daemon/openai.rs` into a folder module

Reason:

- It has the clearest mixed responsibilities
- It improves HTTP extensibility immediately
- It reduces coupling between transport, middleware, metrics, model APIs, and UI serving
- It creates a reusable pattern for the rest of `src/daemon/`

## Tracking

Each major refactor phase should add:

- a short change summary
- updated module tree
- behavioral verification notes
- any follow-up cleanup left intentionally deferred
