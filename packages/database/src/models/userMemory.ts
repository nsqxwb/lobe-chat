import { and, desc, eq } from 'drizzle-orm';

import {
  NewUserMemoryIdentity,
  userMemories,
  userMemoriesContexts,
  userMemoriesExperiences,
  userMemoriesIdentities,
  userMemoriesPreferences,
} from '../schemas/userMemories';
import { LobeChatDatabase } from '../type';

export class UserMemoryModel {
  private userId: string;
  private db: LobeChatDatabase;

  constructor(db: LobeChatDatabase, userId: string) {
    this.userId = userId;
    this.db = db;
  }

  /**
   * Query all user memories
   */
  queryMemories = async () => {
    return this.db.query.userMemories.findMany({
      orderBy: [desc(userMemories.updatedAt)],
      where: eq(userMemories.userId, this.userId),
    });
  };

  /**
   * Count total memories
   */
  countMemories = async () => {
    const result = await this.db
      .select()
      .from(userMemories)
      .where(eq(userMemories.userId, this.userId));

    return result.length;
  };

  /**
   * Query all contexts
   */
  queryContexts = async () => {
    return this.db.query.userMemoriesContexts.findMany({
      orderBy: [desc(userMemoriesContexts.updatedAt)],
    });
  };

  /**
   * Query all preferences
   */
  queryPreferences = async () => {
    return this.db.query.userMemoriesPreferences.findMany({
      orderBy: [desc(userMemoriesPreferences.updatedAt)],
    });
  };

  /**
   * Query all identities
   */
  queryIdentities = async () => {
    return this.db.query.userMemoriesIdentities.findMany({
      orderBy: [desc(userMemoriesIdentities.updatedAt)],
    });
  };

  /**
   * Query all experiences
   */
  queryExperiences = async () => {
    return this.db.query.userMemoriesExperiences.findMany({
      orderBy: [desc(userMemoriesExperiences.updatedAt)],
    });
  };

  /**
   * Find memory by ID
   */
  findById = async (id: string) => {
    return this.db.query.userMemories.findFirst({
      where: and(eq(userMemories.id, id), eq(userMemories.userId, this.userId)),
    });
  };

  // ============ Identity CRUD Operations ============

  /**
   * Create a new identity
   */
  createIdentity = async (data: NewUserMemoryIdentity) => {
    const result = await this.db.insert(userMemoriesIdentities).values(data).returning();
    return result[0];
  };

  /**
   * Update an identity
   */
  updateIdentity = async (id: string, data: Partial<NewUserMemoryIdentity>) => {
    const result = await this.db
      .update(userMemoriesIdentities)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(userMemoriesIdentities.id, id))
      .returning();
    return result[0];
  };

  /**
   * Delete an identity
   */
  deleteIdentity = async (id: string) => {
    await this.db.delete(userMemoriesIdentities).where(eq(userMemoriesIdentities.id, id));
  };

  /**
   * Find identity by ID
   */
  findIdentityById = async (id: string) => {
    return this.db.query.userMemoriesIdentities.findFirst({
      where: eq(userMemoriesIdentities.id, id),
    });
  };
}
