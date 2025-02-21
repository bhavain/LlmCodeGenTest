"use client";

import { useState } from "react";
import { getBenchmarkResults } from "../../lib/api";

export default function Results() {
  const [generationId, setGenerationId] = useState("");
  interface Result {
    submission_id: string;
    test_case_id: string;
    passed: string;
  }

  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const response = await getBenchmarkResults(generationId);
      setResults(response.results);
    } catch (error) {
      console.error("Error fetching results:", error);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-3xl mx-auto mt-10">
      <h1 className="text-3xl font-bold mb-4">Benchmark Results</h1>
      <input
        type="text"
        className="border p-2 w-full mb-4"
        placeholder="Enter Generation ID"
        value={generationId}
        onChange={(e) => setGenerationId(e.target.value)}
      />
      <button onClick={fetchResults} className="bg-blue-500 text-white px-4 py-2">
        {loading ? "Fetching..." : "Fetch Results"}
      </button>

      {results.length > 0 && (
        <table className="mt-4 w-full border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-200">
              <th className="border p-2">Submission ID</th>
              <th className="border p-2">Test Case ID</th>
              <th className="border p-2">Result</th>
            </tr>
          </thead>
          <tbody>
            {results.map((result, index) => (
              <tr key={index} className="text-center">
                <td className="border p-2">{result.submission_id}</td>
                <td className="border p-2">{result.test_case_id}</td>
                <td className={`border p-2 ${result.passed === "Passed" ? "text-green-500" : "text-red-500"}`}>
                  {result.passed}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
