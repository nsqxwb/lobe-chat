import { shallow } from 'zustand/shallow';
import { createWithEqualityFn } from 'zustand/traditional';
import { StateCreator } from 'zustand/vanilla';

import { createDevtools } from '../middleware/createDevtools';
import { UserMemoryStoreState, initialState } from './initialState';
import { MemoryAction, createMemorySlice } from './slices/memory';

//  ===============  聚合 createStoreFn ============ //

export type UserMemoryStore = UserMemoryStoreState & MemoryAction;

const createStore: StateCreator<UserMemoryStore, [['zustand/devtools', never]]> = (
  ...parameters
) => ({
  ...initialState,
  ...createMemorySlice(...parameters),
});

//  ===============  实装 useStore ============ //
const devtools = createDevtools('userMemory');

export const useUserMemoryStore = createWithEqualityFn<UserMemoryStore>()(
  devtools(createStore),
  shallow,
);

export const getUserMemoryStoreState = () => useUserMemoryStore.getState();
