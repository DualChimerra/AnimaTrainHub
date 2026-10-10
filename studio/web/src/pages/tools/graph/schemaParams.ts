import type { TFunction } from 'i18next'
import type { SchemaProperty, SchemaResponse } from '../../../api/client'
import { controlKind, schemaDescription, schemaEnumLabel, schemaFieldLabel, schemaGroupLabel } from '../../../lib/schema'
import { newId } from './model'
import type { Param, ParamOption } from './types'

/** Use the training form's schema and group order, including advanced fields. */
export function schemaParamGroups(schema: SchemaResponse, t: TFunction) {
  const fields = Object.entries(schema.schema.properties).filter(([, p]) => !p.hidden)
  const groups = [...schema.groups]
  for (const [, p] of fields) {
    const key = p.group ?? 'misc'
    if (!groups.some((g) => g.key === key)) groups.push({ key, label: key })
  }
  return groups.map((g) => ({
    key: g.key,
    label: schemaGroupLabel(g.key, g.label, t),
    fields: fields.filter(([, p]) => (p.group ?? 'misc') === g.key),
  })).filter((g) => g.fields.length)
}

/** Create a linked parameter so importing and creating trainings work immediately. */
export function paramFromSchema(key: string, property: SchemaProperty, t: TFunction): Param {
  const kind = controlKind(property)
  const base = {
    id: newId('p'), name: schemaFieldLabel(key, t),
    description: schemaDescription(key, property.description, t),
  }
  if (kind === 'select') {
    return { ...base, type: 'list', options: (property.enum ?? []).map((value) => ({
      id: newId('o'), label: schemaEnumLabel(key, value, t), config: { apply: { [key]: value } },
    })) }
  }
  if (kind === 'bool' || kind === 'tristate') {
    const options: ParamOption[] = [
      { id: 'yes', label: t('graph.yes'), config: { apply: { [key]: true } } },
      { id: 'no', label: t('graph.no'), config: { apply: { [key]: false } } },
    ]
    if (kind === 'tristate') options.push({
      id: 'auto', label: t('graph.auto'), config: { apply: { [key]: null } },
    })
    return { ...base, type: 'bool', options }
  }
  if (kind === 'int' || kind === 'float' || kind === 'int-list' || kind === 'float-list') {
    const list = kind.endsWith('-list')
    const defaults = Array.isArray(property.default) ? property.default : [property.default]
    return { ...base, type: 'number', config: { key, ...(list ? { list: true } : {}) },
      options: defaults.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
        .map((value) => ({ id: newId('o'), value })),
    }
  }
  const json = property.type === 'array' || property.type === 'object' || kind === 'string-list'
  return { ...base, type: 'text', options: [], config: { key, ...(json ? { json: true } : {}) } }
}
