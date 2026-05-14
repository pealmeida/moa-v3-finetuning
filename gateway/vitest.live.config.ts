import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/live-ollama.test.ts'],
    testTimeout: 120000,
  },
});
