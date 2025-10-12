import Papa from 'papaparse';

type FetchOptions = {
  optional?: boolean;
};

async function fetchText(path: string, options?: FetchOptions): Promise<string> {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    if (options?.optional && response.status === 404) {
      return '';
    }
    throw new Error(`Failed to fetch ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

export async function fetchCsv<T>(path: string, options?: FetchOptions): Promise<T[]> {
  const text = await fetchText(path, options);
  if (!text) {
    return [];
  }
  const parsed = Papa.parse<T>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    transformHeader: (header: string) => header.trim(),
  });
  return parsed.data as T[];
}

export async function fetchJsonl<T>(path: string, options?: FetchOptions): Promise<T[]> {
  const text = await fetchText(path, options);
  if (!text) {
    return [];
  }
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}
