"use client";

import { useState } from "react";
import { generateTestCases, runBenchmark } from "../../lib/api";

export default function Test() {
  const [problemIds, setProblemIds] = useState("");
  const [generationId, setGenerationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleGenerateTests = async () => {
    setLoading(true);
    try {
      const response = await generateTestCases(problemIds.split(","));
      setGenerationId(response.generation_id);
    } catch (error) {
      console.error("Error generating test cases:", error);
    }
    setLoading(false);
  };

  const handleRunBenchmark = async () => {
    setLoading(true);
    try {
        const response = await runBenchmark(problemIds.split(","));
        setGenerationId(response.generation_id);
    } catch (error) {
      console.error("Error running benchmark:", error);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-3xl mx-auto mt-10">
      <h1 className="text-3xl font-bold mb-4">Test Case Generation</h1>
      <input
        type="text"
        className="border p-2 w-full mb-4 text-black"
        placeholder="Enter problem IDs (comma-separated)"

        value={problemIds}
        onChange={(e) => setProblemIds(e.target.value)}
      />
      <button onClick={handleGenerateTests} className="bg-blue-500 text-white px-4 py-2">
        {loading ? "Generating..." : "Generate Test Cases"}
      </button>

      {generationId && <p>Generation ID: {generationId}</p>}

      <div className="mt-4">
        <button onClick={handleRunBenchmark} className="bg-green-500 text-white px-4 py-2 mt-2">
          {loading ? "Running..." : "Run Benchmark"}
        </button>
      </div>
    </div>
  );
}
