import { expect, test } from "@playwright/test";

// End-to-end golden path:
//
//   onboarding → workspace (mock) → embedding service → chunking
//   service → knowledge base → upsert two records via API → playground
//   vector query → hits.
//
// We deliberately stay on the vector lane: text retrieval would build
// an `Embedder` under the hood, and `embedding.provider: "mock"`
// throws `embedding_unavailable` from the production embedder factory.
// Vector input bypasses the embedder entirely and is enough to prove
// the end-to-end shell works.
//
// The runtime is memory-backed (default workbench.yaml). State does
// not persist between specs.

test.describe.configure({ mode: "serial" });

test("golden path: onboard → services → knowledge base → upsert → run query", async ({
	page,
	request,
}) => {
	// 1. Visit root. With no workspaces, redirects to /onboarding.
	await page.goto("/");
	await expect(page).toHaveURL(/\/onboarding/);
	await expect(
		page.getByRole("heading", { name: /Manage AI-ready data at scale/ }),
	).toBeVisible();

	// 2. Pick Mock, then proceed to details.
	await page.getByRole("button", { name: /Mock/ }).click();
	await page.getByRole("button", { name: "Continue" }).click();
	await expect(
		page.getByRole("heading", { name: "Workspace details" }),
	).toBeVisible();

	// 3. Fill workspace details. Mock kind needs no credentials.
	await page.getByLabel("Name").fill("e2e-golden");
	await page.getByRole("button", { name: "Create workspace" }).click();

	// 4. Land on workspace detail; capture UID for API calls.
	await expect(page).toHaveURL(/\/workspaces\/[0-9a-f-]{36}/);
	const workspaceUid = page.url().split("/").pop() as string;
	await expect(page.getByRole("heading", { name: "e2e-golden" })).toBeVisible();

	// 5. Create the chunking + embedding services + knowledge base via
	//    API. The UI flow for these is a multi-dialog walk that's
	//    covered by component-level tests; here we just need a
	//    KB to query against.
	const emb = await request
		.post(`/api/v1/workspaces/${workspaceUid}/embedding-services`, {
			data: {
				name: "mock-embedder",
				provider: "mock",
				modelName: "mock-embedder",
				embeddingDimension: 4,
			},
		})
		.then((r) => r.json());
	const chunk = await request
		.post(`/api/v1/workspaces/${workspaceUid}/chunking-services`, {
			data: { name: "default-chunker", engine: "docling" },
		})
		.then((r) => r.json());
	const kb = await request
		.post(`/api/v1/workspaces/${workspaceUid}/knowledge-bases`, {
			data: {
				name: "kb",
				embeddingServiceId: emb.embeddingServiceId,
				chunkingServiceId: chunk.chunkingServiceId,
			},
		})
		.then((r) => r.json());
	const knowledgeBaseUid = kb.knowledgeBaseId as string;

	// 6. Drop straight to the data-plane upsert endpoint — direct
	//    upsert is the contract we're proving here.
	const upsert = await request.post(
		`/api/v1/workspaces/${workspaceUid}/knowledge-bases/${knowledgeBaseUid}/records`,
		{
			data: {
				records: [
					{ id: "alpha", vector: [1, 0, 0, 0], payload: { tag: "keep" } },
					{
						id: "bravo",
						vector: [0.9, 0.1, 0, 0],
						payload: { tag: "keep" },
					},
				],
			},
		},
	);
	expect(upsert.ok()).toBe(true);

	// 7. Navigate to playground.
	await page.getByRole("link", { name: "Playground", exact: true }).click();
	await expect(page.getByRole("heading", { name: "Playground" })).toBeVisible();

	// 8. Pick workspace + KB from the two Radix selects.
	await page.getByLabel("Workspace").click();
	await page.getByRole("option", { name: /e2e-golden/ }).click();
	await page.getByLabel("Knowledge base").click();
	await page.getByRole("option", { name: /kb/ }).click();

	// 9. Switch to the Vector tab and paste a matching vector.
	await page.getByRole("button", { name: "Vector", exact: true }).click();
	await page.getByLabel(/Vector \(/).fill("[1, 0, 0, 0]");

	// 10. Run the query — both rows visible.
	await page.getByRole("button", { name: /Run query/ }).click();
	await expect(page.getByText(/alpha/, { exact: false })).toBeVisible();
	await expect(page.getByText(/bravo/, { exact: false })).toBeVisible();
});
