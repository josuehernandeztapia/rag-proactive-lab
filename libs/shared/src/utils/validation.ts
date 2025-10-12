import { z } from 'zod';

// Common validation schemas
export const PaginationSchema = z.object({
  page: z.number().min(1).default(1),
  limit: z.number().min(1).max(100).default(20),
  search: z.string().optional(),
  sort: z.string().optional(),
  order: z.enum(['asc', 'desc']).default('desc'),
});

export const PlacaSchema = z.string().regex(/^[A-Z0-9]{6,8}$/, 'Invalid placa format');

export const DateRangeSchema = z.object({
  startDate: z.string().datetime(),
  endDate: z.string().datetime(),
}).refine(
  (data) => new Date(data.startDate) <= new Date(data.endDate),
  { message: 'Start date must be before end date' }
);

// Validation helpers
export function validatePlaca(placa: string): boolean {
  return PlacaSchema.safeParse(placa).success;
}

export function sanitizeSearch(search?: string): string | undefined {
  if (!search) return undefined;
  return search.trim().toLowerCase();
}