/**
 * Unit tests for useLocalStorageState.
 *
 * Covers:
 *   - Initial value: falls back to default when localStorage is empty, otherwise
 *     uses the parsed stored value
 *   - setter: writing to localStorage synchronously updates state
 *   - functional setter
 *   - 'storage' event syncing across tabs: a new value syncs into state, and
 *     null (deleted by another tab) reverts to default
 *   - parse failure falls back to default
 */
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { useLocalStorageState } from './useLocalStorageState'

afterEach(() => {
  window.localStorage.clear()
})

describe('useLocalStorageState', () => {
  it('returns defaultValue when storage is empty', () => {
    const { result } = renderHook(() => useLocalStorageState('k', 42))
    expect(result.current[0]).toBe(42)
  })

  it('reads existing stored value at mount', () => {
    window.localStorage.setItem('k', JSON.stringify('hello'))
    const { result } = renderHook(() => useLocalStorageState('k', 'default'))
    expect(result.current[0]).toBe('hello')
  })

  it('writes to localStorage on setValue', () => {
    const { result } = renderHook(() => useLocalStorageState<number>('k', 1))
    act(() => result.current[1](7))
    expect(result.current[0]).toBe(7)
    expect(window.localStorage.getItem('k')).toBe('7')
  })

  it('supports functional updater', () => {
    const { result } = renderHook(() => useLocalStorageState<number>('k', 10))
    act(() => result.current[1]((prev) => prev + 5))
    expect(result.current[0]).toBe(15)
  })

  it('syncs from storage event (other tab wrote new value)', () => {
    const { result } = renderHook(() => useLocalStorageState('k', 'a'))
    act(() => {
      window.dispatchEvent(
        new StorageEvent('storage', { key: 'k', newValue: JSON.stringify('b') }),
      )
    })
    expect(result.current[0]).toBe('b')
  })

  it('reverts to default when storage event signals delete (newValue=null)', () => {
    window.localStorage.setItem('k', JSON.stringify('x'))
    const { result } = renderHook(() => useLocalStorageState('k', 'fallback'))
    expect(result.current[0]).toBe('x')
    act(() => {
      window.dispatchEvent(new StorageEvent('storage', { key: 'k', newValue: null }))
    })
    expect(result.current[0]).toBe('fallback')
  })

  it('ignores storage event for unrelated keys', () => {
    const { result } = renderHook(() => useLocalStorageState('k', 'a'))
    act(() => {
      window.dispatchEvent(
        new StorageEvent('storage', { key: 'other', newValue: JSON.stringify('b') }),
      )
    })
    expect(result.current[0]).toBe('a')
  })

  it('falls back to default when stored value is unparsable', () => {
    window.localStorage.setItem('k', '{not json')
    const { result } = renderHook(() => useLocalStorageState('k', 'fallback'))
    expect(result.current[0]).toBe('fallback')
  })

  it('works with object values', () => {
    type Pref = { mode: 'dark' | 'light'; size: number }
    const { result } = renderHook(() =>
      useLocalStorageState<Pref>('k', { mode: 'light', size: 12 }),
    )
    act(() => result.current[1]({ mode: 'dark', size: 14 }))
    expect(result.current[0]).toEqual({ mode: 'dark', size: 14 })
    expect(JSON.parse(window.localStorage.getItem('k')!)).toEqual({
      mode: 'dark',
      size: 14,
    })
  })
})
