/**
 * Adapters — Unified exports for all execution backends.
 *
 * Usage:
 *   import { OllamaAdapter, LocalAdapter, CloudApiAdapter } from 'moa-gateway-router/adapters';
 */
export { LocalAdapter } from './local-adapter.js';
export { CloudApiAdapter } from './cloud-api-adapter.js';
export { OllamaAdapter } from './ollama-adapter.js';
export { CliAdapter } from './cli-adapter.js';
export { AdapterRegistry } from './registry.js';
export type { ModelAdapter, AdapterConfig, GenerateRequest, GenerateChunk, GenerateResult, TokenUsage, ExecutionBackend, AdapterEntry } from './types.js';
