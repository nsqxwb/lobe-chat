import { ModelUsage } from '@lobechat/types';

import { AgentState } from '../types';

/**
 * UsageCounter - Centralized usage and cost accumulation for AgentState
 * Encapsulates all logic for tracking LLM and tool usage/costs
 *
 * Immutable design: All methods return updated state.
 */
/* eslint-disable unicorn/no-static-only-class */
export class UsageCounter {
  /**
   * Merge two ModelUsage objects by accumulating token counts
   * @param previous - Previous usage statistics
   * @param current - Current usage statistics to add
   * @returns Merged usage statistics
   */
  private static mergeModelUsage(
    previous: ModelUsage | undefined,
    current: ModelUsage,
  ): ModelUsage {
    if (!previous) return current;

    const merged: ModelUsage = { ...current };

    // Accumulate all numeric token fields
    const numericFields: (keyof ModelUsage)[] = [
      'inputCachedTokens',
      'inputCacheMissTokens',
      'inputWriteCacheTokens',
      'inputTextTokens',
      'inputImageTokens',
      'inputAudioTokens',
      'inputCitationTokens',
      'outputTextTokens',
      'outputImageTokens',
      'outputAudioTokens',
      'outputReasoningTokens',
      'acceptedPredictionTokens',
      'rejectedPredictionTokens',
      'totalInputTokens',
      'totalOutputTokens',
      'totalTokens',
    ];

    for (const field of numericFields) {
      const prevValue = previous[field] as number | undefined;
      const currValue = current[field] as number | undefined;

      if (prevValue !== undefined || currValue !== undefined) {
        merged[field] = (prevValue || 0) + (currValue || 0);
      }
    }

    // Accumulate cost
    if (previous.cost !== undefined || current.cost !== undefined) {
      merged.cost = (previous.cost || 0) + (current.cost || 0);
    }

    return merged;
  }
  /**
   * Accumulate LLM usage and cost for a specific model
   * @param state - Current agent state
   * @param provider - Provider name (e.g., "openai")
   * @param model - Model name (e.g., "gpt-4")
   * @param usage - ModelUsage from model-runtime
   * @returns Updated AgentState with accumulated usage and cost
   */
  static accumulateLLM(
    state: AgentState,
    provider: string,
    model: string,
    usage: ModelUsage,
  ): AgentState {
    const newState = structuredClone(state);

    // 1. Accumulate token counts to usage.llm
    if (!newState.usage) {
      throw new Error('AgentState.usage is not initialized');
    }

    newState.usage.llm.tokens.input += usage.totalInputTokens ?? 0;
    newState.usage.llm.tokens.output += usage.totalOutputTokens ?? 0;
    newState.usage.llm.tokens.total += usage.totalTokens ?? 0;
    newState.usage.llm.apiCalls += 1;

    // 2. Accumulate cost (only when cost is available)
    if (usage.cost) {
      if (!newState.cost) {
        throw new Error('AgentState.cost is not initialized');
      }

      const modelId = `${provider}/${model}`;

      // Find or create byModel entry
      let modelEntry = newState.cost.llm.byModel.find((entry) => entry.id === modelId);

      if (!modelEntry) {
        modelEntry = {
          id: modelId,
          model,
          provider,
          totalCost: 0,
          usage: {},
        };
        newState.cost.llm.byModel.push(modelEntry);
      }

      // Merge usage breakdown
      modelEntry.usage = UsageCounter.mergeModelUsage(modelEntry.usage, usage);

      // Accumulate costs
      modelEntry.totalCost += usage.cost;
      newState.cost.llm.total += usage.cost;
      newState.cost.total += usage.cost;
      newState.cost.calculatedAt = new Date().toISOString();
    }

    return newState;
  }

  /**
   * Accumulate tool usage and cost
   * @param state - Current agent state
   * @param toolName - Tool identifier
   * @param executionTime - Execution time in milliseconds
   * @param success - Whether the execution was successful
   * @param cost - Optional cost for this tool call
   * @returns Updated AgentState with accumulated tool usage and cost
   */
  static accumulateTool(
    state: AgentState,
    toolName: string,
    executionTime: number,
    success: boolean,
    cost?: number,
  ): AgentState {
    const newState = structuredClone(state);

    if (!newState.usage) {
      throw new Error('AgentState.usage is not initialized');
    }

    // Find or create byTool entry
    let toolEntry = newState.usage.tools.byTool.find((entry) => entry.name === toolName);

    if (!toolEntry) {
      toolEntry = {
        calls: 0,
        errors: 0,
        name: toolName,
        totalTimeMs: 0,
      };
      newState.usage.tools.byTool.push(toolEntry);
    }

    // Accumulate tool usage
    toolEntry.calls += 1;
    toolEntry.totalTimeMs += executionTime;
    if (!success) {
      toolEntry.errors += 1;
    }

    newState.usage.tools.totalCalls += 1;
    newState.usage.tools.totalTimeMs += executionTime;

    // Accumulate cost if provided
    if (cost && newState.cost) {
      let toolCostEntry = newState.cost.tools.byTool.find((entry) => entry.name === toolName);

      if (!toolCostEntry) {
        toolCostEntry = {
          calls: 0,
          currency: 'USD',
          name: toolName,
          totalCost: 0,
        };
        newState.cost.tools.byTool.push(toolCostEntry);
      }

      toolCostEntry.calls += 1;
      toolCostEntry.totalCost += cost;
      newState.cost.tools.total += cost;
      newState.cost.total += cost;
      newState.cost.calculatedAt = new Date().toISOString();
    }

    return newState;
  }
}
