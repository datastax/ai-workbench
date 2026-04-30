/**
 * Tests for the agent tool registry. The fixtures lean on the
 * existing memory control plane + mock vector driver so we exercise
 * the real store / driver wiring rather than mocking it out — the
 * tools are thin wrappers and most of the value here is verifying
 * we plumb the right ids through and shape the JSON the LLM sees.
 */

import { randomUUID } from "node:crypto";
import { describe, expect, test } from "vitest";
import {
	type AgentToolDeps,
	DEFAULT_AGENT_TOOLS,
	resolveTool,
} from "../../src/chat/tools/registry.js";
import { MemoryControlPlaneStore } from "../../src/control-plane/memory/store.js";
import { MockVectorStoreDriver } from "../../src/drivers/mock/store.js";
import { VectorStoreDriverRegistry } from "../../src/drivers/registry.js";
import { makeFakeEmbedderFactory } from "../helpers/embedder.js";

interface Fixture {
	deps: AgentToolDeps;
	store: MemoryControlPlaneStore;
	workspaceId: string;
}

async function fixture(): Promise<Fixture> {
	const store = new MemoryControlPlaneStore();
	const driver = new MockVectorStoreDriver();
	const drivers = new VectorStoreDriverRegistry(new Map([["mock", driver]]));
	const embedders = makeFakeEmbedderFactory();
	const ws = await store.createWorkspace({ name: "ws", kind: "mock" });
	return {
		deps: {
			workspaceId: ws.uid,
			store,
			drivers,
			embedders,
		},
		store,
		workspaceId: ws.uid,
	};
}

async function seedKb(
	store: MemoryControlPlaneStore,
	workspaceId: string,
	name: string,
): Promise<string> {
	const chunk = await store.createChunkingService(workspaceId, {
		name: `chunk-${name}`,
		engine: "langchain_ts",
	});
	const emb = await store.createEmbeddingService(workspaceId, {
		name: `emb-${name}`,
		provider: "fake",
		modelName: "fake",
		embeddingDimension: 8,
	});
	const kb = await store.createKnowledgeBase(workspaceId, {
		name,
		chunkingServiceId: chunk.chunkingServiceId,
		embeddingServiceId: emb.embeddingServiceId,
	});
	return kb.knowledgeBaseId;
}

describe("agent tool registry", () => {
	test("DEFAULT_AGENT_TOOLS exposes the canonical tool set", () => {
		const names = DEFAULT_AGENT_TOOLS.map((t) => t.definition.name).sort();
		expect(names).toEqual([
			"count_documents",
			"get_document",
			"list_documents",
			"list_kbs",
			"search_kb",
			"summarize_kb",
		]);
		// Every definition has a non-empty description and a JSON Schema
		// that the LLM can parse — the actual call shape is locked down
		// by the OpenAIChatService tests.
		for (const t of DEFAULT_AGENT_TOOLS) {
			expect(t.definition.description.length).toBeGreaterThan(20);
			expect(t.definition.parameters).toMatchObject({ type: "object" });
		}
	});

	test("resolveTool returns null for unknown names", () => {
		expect(resolveTool("definitely_not_a_tool")).toBeNull();
		expect(resolveTool("search_kb")).not.toBeNull();
	});
});

describe("list_kbs", () => {
	test("returns 'no knowledge bases' when the workspace is empty", async () => {
		const f = await fixture();
		const tool = resolveTool("list_kbs");
		const out = await tool?.execute({}, f.deps);
		expect(out).toMatch(/no knowledge bases/i);
	});

	test("lists each KB's id, name, description", async () => {
		const f = await fixture();
		await seedKb(f.store, f.workspaceId, "alpha");
		await seedKb(f.store, f.workspaceId, "beta");
		const out = await resolveTool("list_kbs")?.execute({}, f.deps);
		const parsed = JSON.parse(out as string);
		expect(parsed.knowledgeBases).toHaveLength(2);
		const names = parsed.knowledgeBases
			.map((k: { name: string }) => k.name)
			.sort();
		expect(names).toEqual(["alpha", "beta"]);
	});
});

describe("count_documents", () => {
	test("counts zero across an empty workspace", async () => {
		const f = await fixture();
		const out = await resolveTool("count_documents")?.execute({}, f.deps);
		const parsed = JSON.parse(out as string);
		expect(parsed.total).toBe(0);
		expect(parsed.perKnowledgeBase).toEqual([]);
	});

	test("aggregates across all KBs when knowledgeBaseId is omitted", async () => {
		const f = await fixture();
		const kb1 = await seedKb(f.store, f.workspaceId, "alpha");
		const kb2 = await seedKb(f.store, f.workspaceId, "beta");
		await f.store.createRagDocument(f.workspaceId, kb1, {
			sourceFilename: "a.txt",
		});
		await f.store.createRagDocument(f.workspaceId, kb1, {
			sourceFilename: "b.txt",
		});
		await f.store.createRagDocument(f.workspaceId, kb2, {
			sourceFilename: "c.txt",
		});

		const out = await resolveTool("count_documents")?.execute({}, f.deps);
		const parsed = JSON.parse(out as string);
		expect(parsed.total).toBe(3);
		expect(parsed.perKnowledgeBase).toHaveLength(2);
	});
});

describe("summarize_kb", () => {
	test("returns sample document filenames + counts", async () => {
		const f = await fixture();
		const kb = await seedKb(f.store, f.workspaceId, "main");
		for (let i = 0; i < 7; i++) {
			await f.store.createRagDocument(f.workspaceId, kb, {
				sourceFilename: `doc-${i}.md`,
			});
		}
		const out = await resolveTool("summarize_kb")?.execute({}, f.deps);
		const parsed = JSON.parse(out as string);
		expect(parsed.summaries).toHaveLength(1);
		expect(parsed.summaries[0].documentCount).toBe(7);
		expect(parsed.summaries[0].sampleDocuments).toHaveLength(5);
		expect(parsed.summaries[0].sampleDocuments[0]).toMatchObject({
			sourceFilename: expect.stringMatching(/^doc-\d\.md$/),
		});
	});

	test("returns a clean error when knowledgeBaseId is unknown", async () => {
		const f = await fixture();
		const out = await resolveTool("summarize_kb")?.execute(
			{ knowledgeBaseId: randomUUID() },
			f.deps,
		);
		expect(out).toMatch(/^Error:/);
	});
});

describe("validation", () => {
	test("rejects unknown args fields with a clean Error: string", async () => {
		const f = await fixture();
		const out = await resolveTool("count_documents")?.execute(
			{ knowledgeBaseId: "not-a-uuid" },
			f.deps,
		);
		expect(out).toMatch(/^Error:/);
	});

	test("search_kb rejects missing query with a clean error string", async () => {
		const f = await fixture();
		const out = await resolveTool("search_kb")?.execute({}, f.deps);
		expect(out).toMatch(/^Error:/);
	});
});
