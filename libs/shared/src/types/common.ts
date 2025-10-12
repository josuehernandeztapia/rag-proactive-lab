// Common types shared across all agents and clients

export interface BaseResponse {
  success: boolean;
  message?: string;
  timestamp: string;
}

export interface ErrorResponse extends BaseResponse {
  success: false;
  error: string;
  code?: string;
}

export interface SuccessResponse<T = any> extends BaseResponse {
  success: true;
  data: T;
}

export type ApiResponse<T = any> = SuccessResponse<T> | ErrorResponse;

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

export interface QueryParams {
  page?: number;
  limit?: number;
  search?: string;
  sort?: string;
  order?: 'asc' | 'desc';
}