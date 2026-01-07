/**
 * MongoDB Initialization Script
 * 
 * This script runs when the MongoDB container is first initialized.
 * It creates the necessary collections and indexes for the AI Agents Platform.
 */

// Switch to the ai_agents database
db = db.getSiblingDB('ai_agents');

// Create sessions collection with schema validation
db.createCollection('sessions', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['_id', 'user_id', 'status', 'created_at', 'updated_at'],
            properties: {
                _id: {
                    bsonType: 'string',
                    description: 'Session UUID as string'
                },
                user_id: {
                    bsonType: 'string',
                    description: 'User identifier'
                },
                agent_id: {
                    bsonType: ['string', 'null'],
                    description: 'Agent UUID as string'
                },
                status: {
                    enum: ['active', 'paused', 'expired', 'terminated'],
                    description: 'Session status'
                },
                context: {
                    bsonType: 'object',
                    description: 'Session context data'
                },
                metadata: {
                    bsonType: 'object',
                    description: 'Session metadata'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Update timestamp'
                },
                expires_at: {
                    bsonType: ['date', 'null'],
                    description: 'Expiration timestamp'
                }
            }
        }
    }
});

// Create indexes for sessions collection
db.sessions.createIndex({ 'user_id': 1 });
db.sessions.createIndex({ 'status': 1 });
db.sessions.createIndex({ 'created_at': -1 });
db.sessions.createIndex({ 'expires_at': 1 }, { expireAfterSeconds: 0 });

// Create checkpoints collection for LangGraph
db.createCollection('checkpoints');

// Create indexes for checkpoints collection
db.checkpoints.createIndex({ 'thread_id': 1 });
db.checkpoints.createIndex({ 'checkpoint_id': 1 });
db.checkpoints.createIndex({ 'thread_id': 1, 'checkpoint_id': 1 }, { unique: true });

// Create conversations collection (optional, for persistence)
db.createCollection('conversations');
db.conversations.createIndex({ 'session_id': 1 });
db.conversations.createIndex({ 'created_at': -1 });

print('MongoDB initialization completed successfully!');
print('Collections created: sessions, checkpoints, conversations');
