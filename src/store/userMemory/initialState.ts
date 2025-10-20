import { UserMemoryItem } from '@lobechat/types';

export interface UserMemoryStoreState {
  memories: UserMemoryItem[];
}

export const initialState: UserMemoryStoreState = {
  memories: [],
};
