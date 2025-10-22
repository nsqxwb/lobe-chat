import { ModelUsage } from '@lobechat/types';
import { describe, expect, it } from 'vitest';

import { AgentState } from '../../types';
import { UsageCounter } from '../UsageCounter';
import { AgentRuntime } from '../runtime';

describe('UsageCounter', () => {
  describe('UsageCounter.accumulateLLM', () => {
    it('should accumulate LLM usage tokens', () => {
      const state = AgentRuntime.createInitialState();

      const usage: ModelUsage = {
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const newState = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage);

      expect(newState.usage.llm.tokens.input).toBe(100);
      expect(newState.usage.llm.tokens.output).toBe(50);
      expect(newState.usage.llm.tokens.total).toBe(150);
      expect(newState.usage.llm.apiCalls).toBe(1);
    });

    it('should not mutate original state', () => {
      const state = AgentRuntime.createInitialState();

      const usage: ModelUsage = {
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const newState = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage);

      expect(state.usage.llm.tokens.input).toBe(0);
      expect(newState).not.toBe(state);
    });

    it('should create new byModel entry when not exists', () => {
      const state = AgentRuntime.createInitialState();

      const usage: ModelUsage = {
        cost: 0.05,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const newState = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage);

      expect(newState.cost.llm.byModel).toHaveLength(1);
      expect(newState.cost.llm.byModel[0]).toEqual({
        id: 'openai/gpt-4',
        model: 'gpt-4',
        provider: 'openai',
        totalCost: 0.05,
        usage: {
          cost: 0.05,
          totalInputTokens: 100,
          totalOutputTokens: 50,
          totalTokens: 150,
        },
      });
    });

    it('should accumulate to existing byModel entry', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const usage2: ModelUsage = {
        cost: 0.03,
        totalInputTokens: 50,
        totalOutputTokens: 25,
        totalTokens: 75,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage1);
      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage2);

      expect(state.cost.llm.byModel).toHaveLength(1);
      expect(state.cost.llm.byModel[0]).toEqual({
        id: 'openai/gpt-4',
        model: 'gpt-4',
        provider: 'openai',
        totalCost: 0.08,
        usage: {
          cost: 0.08,
          totalInputTokens: 150,
          totalOutputTokens: 75,
          totalTokens: 225,
        },
      });
    });

    it('should accumulate multiple models separately', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const usage2: ModelUsage = {
        cost: 0.02,
        totalInputTokens: 50,
        totalOutputTokens: 25,
        totalTokens: 75,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage1);
      state = UsageCounter.accumulateLLM(state, 'anthropic', 'claude-3-5-sonnet-20241022', usage2);

      expect(state.cost.llm.byModel).toHaveLength(2);
      expect(state.cost.llm.byModel[0].id).toBe('openai/gpt-4');
      expect(state.cost.llm.byModel[1].id).toBe('anthropic/claude-3-5-sonnet-20241022');
    });

    it('should accumulate cache-related tokens', () => {
      const state = AgentRuntime.createInitialState();

      const usage: ModelUsage = {
        cost: 0.05,
        inputCacheMissTokens: 60,
        inputCachedTokens: 40,
        inputWriteCacheTokens: 20,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const newState = UsageCounter.accumulateLLM(
        state,
        'anthropic',
        'claude-3-5-sonnet-20241022',
        usage,
      );

      expect(newState.cost.llm.byModel[0].usage).toEqual({
        cost: 0.05,
        inputCacheMissTokens: 60,
        inputCachedTokens: 40,
        inputWriteCacheTokens: 20,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      });
    });

    it('should accumulate total costs correctly', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const usage2: ModelUsage = {
        cost: 0.03,
        totalInputTokens: 50,
        totalOutputTokens: 25,
        totalTokens: 75,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage1);
      state = UsageCounter.accumulateLLM(state, 'anthropic', 'claude-3-5-sonnet-20241022', usage2);

      expect(state.cost.llm.total).toBe(0.08);
      expect(state.cost.total).toBe(0.08);
      expect(state.cost.calculatedAt).toBeDefined();
    });

    it('should not accumulate cost when usage.cost is undefined', () => {
      const state = AgentRuntime.createInitialState();

      const usage: ModelUsage = {
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const newState = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage);

      expect(newState.cost.llm.byModel).toHaveLength(0);
      expect(newState.cost.llm.total).toBe(0);
      expect(newState.cost.total).toBe(0);
    });

    it('should increment apiCalls for each accumulation', () => {
      let state = AgentRuntime.createInitialState();

      const usage: ModelUsage = {
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage);
      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage);
      state = UsageCounter.accumulateLLM(state, 'anthropic', 'claude-3-5-sonnet-20241022', usage);

      expect(state.usage.llm.apiCalls).toBe(3);
    });
  });

  describe('UsageCounter.accumulateTool', () => {
    it('should accumulate tool usage', () => {
      const state = AgentRuntime.createInitialState();

      const newState = UsageCounter.accumulateTool(state, 'search', 1000, true);

      expect(newState.usage.tools.byTool).toHaveLength(1);
      expect(newState.usage.tools.byTool[0]).toEqual({
        calls: 1,
        errors: 0,
        name: 'search',
        totalTimeMs: 1000,
      });
      expect(newState.usage.tools.totalCalls).toBe(1);
      expect(newState.usage.tools.totalTimeMs).toBe(1000);
    });

    it('should not mutate original state', () => {
      const state = AgentRuntime.createInitialState();

      const newState = UsageCounter.accumulateTool(state, 'search', 1000, true);

      expect(state.usage.tools.totalCalls).toBe(0);
      expect(newState).not.toBe(state);
    });

    it('should accumulate errors when success is false', () => {
      const state = AgentRuntime.createInitialState();

      const newState = UsageCounter.accumulateTool(state, 'search', 1000, false);

      expect(newState.usage.tools.byTool[0]).toEqual({
        calls: 1,
        errors: 1,
        name: 'search',
        totalTimeMs: 1000,
      });
    });

    it('should accumulate multiple tool calls', () => {
      let state = AgentRuntime.createInitialState();

      state = UsageCounter.accumulateTool(state, 'search', 1000, true);
      state = UsageCounter.accumulateTool(state, 'search', 500, true);
      state = UsageCounter.accumulateTool(state, 'calculator', 200, false);

      expect(state.usage.tools.byTool).toHaveLength(2);
      expect(state.usage.tools.byTool.find((t) => t.name === 'search')).toEqual({
        calls: 2,
        errors: 0,
        name: 'search',
        totalTimeMs: 1500,
      });
      expect(state.usage.tools.byTool.find((t) => t.name === 'calculator')).toEqual({
        calls: 1,
        errors: 1,
        name: 'calculator',
        totalTimeMs: 200,
      });
      expect(state.usage.tools.totalCalls).toBe(3);
      expect(state.usage.tools.totalTimeMs).toBe(1700);
    });

    it('should accumulate tool cost when provided', () => {
      const state = AgentRuntime.createInitialState();

      const newState = UsageCounter.accumulateTool(state, 'premium-search', 1000, true, 0.01);

      expect(newState.cost.tools.byTool).toHaveLength(1);
      expect(newState.cost.tools.byTool[0]).toEqual({
        calls: 1,
        currency: 'USD',
        name: 'premium-search',
        totalCost: 0.01,
      });
      expect(newState.cost.tools.total).toBe(0.01);
      expect(newState.cost.total).toBe(0.01);
    });

    it('should accumulate tool cost across multiple calls', () => {
      let state = AgentRuntime.createInitialState();

      state = UsageCounter.accumulateTool(state, 'premium-search', 1000, true, 0.01);
      state = UsageCounter.accumulateTool(state, 'premium-search', 500, true, 0.005);

      expect(state.cost.tools.byTool).toHaveLength(1);
      expect(state.cost.tools.byTool[0]).toEqual({
        calls: 2,
        currency: 'USD',
        name: 'premium-search',
        totalCost: 0.015,
      });
      expect(state.cost.tools.total).toBe(0.015);
      expect(state.cost.total).toBe(0.015);
    });

    it('should not accumulate cost when cost is undefined', () => {
      const state = AgentRuntime.createInitialState();

      const newState = UsageCounter.accumulateTool(state, 'free-tool', 1000, true);

      expect(newState.cost.tools.byTool).toHaveLength(0);
      expect(newState.cost.tools.total).toBe(0);
    });
  });

  describe('mixed accumulation', () => {
    it('should accumulate both LLM and tool costs correctly', () => {
      let state = AgentRuntime.createInitialState();

      const llmUsage: ModelUsage = {
        cost: 0.05,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', llmUsage);
      state = UsageCounter.accumulateTool(state, 'premium-search', 1000, true, 0.01);

      expect(state.cost.llm.total).toBe(0.05);
      expect(state.cost.tools.total).toBe(0.01);
      expect(state.cost.total).toBeCloseTo(0.06);
    });
  });

  describe('error handling', () => {
    it('should throw error when usage is not initialized', () => {
      const state = {
        createdAt: new Date().toISOString(),
        lastModified: new Date().toISOString(),
        messages: [],
        sessionId: 'test-session',
        status: 'running',
        stepCount: 0,
      } as unknown as AgentState;

      expect(() => {
        UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', { totalInputTokens: 100 });
      }).toThrow('AgentState.usage is not initialized');
    });

    it('should throw error when cost is not initialized but trying to accumulate cost', () => {
      const state = {
        createdAt: new Date().toISOString(),
        lastModified: new Date().toISOString(),
        messages: [],
        sessionId: 'test-session',
        status: 'running',
        stepCount: 0,
        usage: AgentRuntime.createDefaultUsage(),
      } as unknown as AgentState;

      expect(() => {
        UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', { cost: 0.05, totalInputTokens: 100 });
      }).toThrow('AgentState.cost is not initialized');
    });
  });

  describe('mergeModelUsage (private method tests via accumulateLLM)', () => {
    it('should merge basic token counts', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const usage2: ModelUsage = {
        cost: 0.03,
        totalInputTokens: 200,
        totalOutputTokens: 100,
        totalTokens: 300,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage1);
      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage2);

      expect(state.cost.llm.byModel[0].usage).toEqual({
        cost: 0.08,
        totalInputTokens: 300,
        totalOutputTokens: 150,
        totalTokens: 450,
      });
    });

    it('should merge cache-related tokens', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        inputCacheMissTokens: 30,
        inputCachedTokens: 50,
        inputWriteCacheTokens: 20,
        totalInputTokens: 100,
        totalOutputTokens: 50,
        totalTokens: 150,
      };

      const usage2: ModelUsage = {
        cost: 0.03,
        inputCacheMissTokens: 40,
        inputCachedTokens: 80,
        inputWriteCacheTokens: 30,
        totalInputTokens: 150,
        totalOutputTokens: 75,
        totalTokens: 225,
      };

      state = UsageCounter.accumulateLLM(state, 'anthropic', 'claude-3-5-sonnet-20241022', usage1);
      state = UsageCounter.accumulateLLM(state, 'anthropic', 'claude-3-5-sonnet-20241022', usage2);

      expect(state.cost.llm.byModel[0].usage).toEqual({
        cost: 0.08,
        inputCacheMissTokens: 70,
        inputCachedTokens: 130,
        inputWriteCacheTokens: 50,
        totalInputTokens: 250,
        totalOutputTokens: 125,
        totalTokens: 375,
      });
    });

    it('should merge reasoning tokens', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        outputReasoningTokens: 100,
        outputTextTokens: 200,
        totalInputTokens: 100,
        totalOutputTokens: 300,
        totalTokens: 400,
      };

      const usage2: ModelUsage = {
        cost: 0.03,
        outputReasoningTokens: 50,
        outputTextTokens: 100,
        totalInputTokens: 50,
        totalOutputTokens: 150,
        totalTokens: 200,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'o1', usage1);
      state = UsageCounter.accumulateLLM(state, 'openai', 'o1', usage2);

      expect(state.cost.llm.byModel[0].usage).toEqual({
        cost: 0.08,
        outputReasoningTokens: 150,
        outputTextTokens: 300,
        totalInputTokens: 150,
        totalOutputTokens: 450,
        totalTokens: 600,
      });
    });

    it('should merge audio and image tokens', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        inputAudioTokens: 10,
        inputImageTokens: 20,
        outputAudioTokens: 5,
        outputImageTokens: 15,
        totalInputTokens: 30,
        totalOutputTokens: 20,
        totalTokens: 50,
      };

      const usage2: ModelUsage = {
        cost: 0.03,
        inputAudioTokens: 15,
        inputImageTokens: 25,
        outputAudioTokens: 8,
        outputImageTokens: 12,
        totalInputTokens: 40,
        totalOutputTokens: 20,
        totalTokens: 60,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4o-audio-preview', usage1);
      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4o-audio-preview', usage2);

      expect(state.cost.llm.byModel[0].usage).toEqual({
        cost: 0.08,
        inputAudioTokens: 25,
        inputImageTokens: 45,
        outputAudioTokens: 13,
        outputImageTokens: 27,
        totalInputTokens: 70,
        totalOutputTokens: 40,
        totalTokens: 110,
      });
    });

    it('should merge prediction tokens', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        acceptedPredictionTokens: 50,
        cost: 0.05,
        rejectedPredictionTokens: 10,
        totalInputTokens: 100,
        totalOutputTokens: 60,
        totalTokens: 160,
      };

      const usage2: ModelUsage = {
        acceptedPredictionTokens: 30,
        cost: 0.03,
        rejectedPredictionTokens: 5,
        totalInputTokens: 50,
        totalOutputTokens: 35,
        totalTokens: 85,
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4o', usage1);
      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4o', usage2);

      expect(state.cost.llm.byModel[0].usage).toEqual({
        acceptedPredictionTokens: 80,
        cost: 0.08,
        rejectedPredictionTokens: 15,
        totalInputTokens: 150,
        totalOutputTokens: 95,
        totalTokens: 245,
      });
    });

    it('should handle missing fields gracefully', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        cost: 0.05,
        totalInputTokens: 100,
        // totalOutputTokens is missing
      };

      const usage2: ModelUsage = {
        cost: 0.03,
        totalOutputTokens: 50,
        // totalInputTokens is missing
      };

      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage1);
      state = UsageCounter.accumulateLLM(state, 'openai', 'gpt-4', usage2);

      expect(state.cost.llm.byModel[0].usage).toEqual({
        cost: 0.08,
        totalInputTokens: 100,
        totalOutputTokens: 50,
      });
    });

    it('should merge all fields in a comprehensive scenario', () => {
      let state = AgentRuntime.createInitialState();

      const usage1: ModelUsage = {
        acceptedPredictionTokens: 10,
        cost: 0.05,
        inputAudioTokens: 5,
        inputCacheMissTokens: 40,
        inputCachedTokens: 60,
        inputCitationTokens: 10,
        inputImageTokens: 20,
        inputTextTokens: 100,
        inputWriteCacheTokens: 30,
        outputAudioTokens: 3,
        outputImageTokens: 8,
        outputReasoningTokens: 20,
        outputTextTokens: 50,
        rejectedPredictionTokens: 5,
        totalInputTokens: 200,
        totalOutputTokens: 80,
        totalTokens: 280,
      };

      const usage2: ModelUsage = {
        acceptedPredictionTokens: 5,
        cost: 0.03,
        inputAudioTokens: 3,
        inputCacheMissTokens: 20,
        inputCachedTokens: 30,
        inputCitationTokens: 5,
        inputImageTokens: 10,
        inputTextTokens: 50,
        inputWriteCacheTokens: 15,
        outputAudioTokens: 2,
        outputImageTokens: 4,
        outputReasoningTokens: 10,
        outputTextTokens: 25,
        rejectedPredictionTokens: 2,
        totalInputTokens: 100,
        totalOutputTokens: 40,
        totalTokens: 140,
      };

      state = UsageCounter.accumulateLLM(state, 'anthropic', 'claude-3-5-sonnet-20241022', usage1);
      state = UsageCounter.accumulateLLM(state, 'anthropic', 'claude-3-5-sonnet-20241022', usage2);

      expect(state.cost.llm.byModel[0].usage).toEqual({
        acceptedPredictionTokens: 15,
        cost: 0.08,
        inputAudioTokens: 8,
        inputCacheMissTokens: 60,
        inputCachedTokens: 90,
        inputCitationTokens: 15,
        inputImageTokens: 30,
        inputTextTokens: 150,
        inputWriteCacheTokens: 45,
        outputAudioTokens: 5,
        outputImageTokens: 12,
        outputReasoningTokens: 30,
        outputTextTokens: 75,
        rejectedPredictionTokens: 7,
        totalInputTokens: 300,
        totalOutputTokens: 120,
        totalTokens: 420,
      });
    });
  });
});
