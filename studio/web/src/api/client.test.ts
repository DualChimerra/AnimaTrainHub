import { describe, expect, it } from 'vitest'
import { makeApiError } from './client'

// ADR-0009 Phase 2: makeApiError parses the backend error envelope into ApiError.
describe('makeApiError', () => {
  it('body.error.code -> errors.<code> i18n localization + details interpolation', () => {
    const e = makeApiError(404, 'Not Found', {
      detail: 'Preset "foo" not found',
      error: { code: 'preset.not_found', message: 'Preset "foo" not found', details: { name: 'foo' }, trace_id: 'abc123' },
    })
    expect(e.message).toBe('Preset "foo" not found')
    expect(e.code).toBe('preset.not_found')
    expect(e.traceId).toBe('abc123')
    expect(e.status).toBe(404)
  })

  it('unknown code -> falls back to error.message (defaultValue)', () => {
    const e = makeApiError(400, 'Bad Request', {
      detail: 'something specific',
      error: { code: 'http.400', message: 'something specific', trace_id: 't1' },
    })
    expect(e.message).toBe('something specific')
    expect(e.code).toBe('http.400')
  })

  it('no error envelope, detail is a string -> uses detail as the message (legacy fallback)', () => {
    const e = makeApiError(400, 'Bad Request', { detail: 'legacy message' })
    expect(e.message).toBe('legacy message')
    expect(e.code).toBeUndefined()
  })

  it('structured data in error.details (409 conflict) -> attached to err.detail for the callsite', () => {
    const e = makeApiError(409, 'Conflict', {
      detail: 'Preset "foo" already exists',
      error: {
        code: 'preset.exists',
        message: 'Preset "foo" already exists',
        details: { name: 'foo', config: { a: 1 }, suggested_name: 'foo-2' },
        trace_id: 't2',
      },
    })
    expect(e.message).toBe('Preset "foo" already exists')
    expect(e.detail).toEqual({ name: 'foo', config: { a: 1 }, suggested_name: 'foo-2' })
  })

  it('non-JSON body (null) -> falls back to statusText + header trace', () => {
    const e = makeApiError(500, 'Internal Server Error', null, 'hdr-trace')
    expect(e.message).toBe('500 Internal Server Error')
    expect(e.traceId).toBe('hdr-trace')
  })
})
