/*
The below SQL statements check if the PostgreSQL extension named 'vector' is installed, and if not, it installs it. Then, it creates a schema called 'bedrock_integration' and a table named 'bedrock_kb' within that schema, which includes columns for storing UUID, vector embeddings, text chunks, JSON metadata, country, variety, points, and price. 

Finally, it creates an index on the 'embedding' column using the HNSW (Hierarchical Navigable Small World) algorithm for efficient vector similarity searches.
*/

SELECT extversion FROM pg_extension WHERE extname='vector';
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname='vector';
CREATE SCHEMA IF NOT EXISTS bedrock_integration;
CREATE TABLE bedrock_integration.bedrock_kb (id uuid PRIMARY KEY, embedding vector(1024), chunks text, metadata json, country text, variety text, points smallint, price numeric);
CREATE INDEX on bedrock_integration.bedrock_kb USING hnsw (embedding vector_cosine_ops);