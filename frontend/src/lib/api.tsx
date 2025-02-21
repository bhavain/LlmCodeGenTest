import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

export const generateTestCases = async (problemIds: string[]) => {
  const response = await axios.post(`${API_BASE_URL}/testcases/generate`, {
    problem_ids: problemIds,
  },
  {
      headers: {
        "Content-Type": "application/json",
      },
  });
  return response.data;
};

export const runBenchmark = async (problemIds: string[]) => {
  const response = await axios.post(`${API_BASE_URL}/benchmark`, {
    problem_ids: problemIds,
  });
  return response.data;
};

export const getBenchmarkResults = async (generationId: string) => {
  const response = await axios.get(`${API_BASE_URL}/benchmark/${generationId}/results`);
  return response.data;
};
