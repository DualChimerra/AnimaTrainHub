/** Highlights a YAML dump line by line: keys, strings, comments. */
export default function YamlLines({ text }: { text: string }) {
  return (
    <>
      {text.split('\n').map((line, i) => {
        const comment = line.match(/^(\s*)(#.*)$/)
        if (comment) return <div key={i}>{comment[1]}<span className="ds-c">{comment[2]}</span></div>
        const kv = line.match(/^(\s*-?\s*)([^:#'"\s][^:]*?):(\s+(.*))?$/)
        if (!kv) return <div key={i}>{line || ' '}</div>
        const value = kv[4] ?? ''
        const isString = /^['"]/.test(value) || (value !== '' && !/^(-?[\d.e+-]+|true|false|null|~|\[.*\]|\{.*\})$/i.test(value))
        return (
          <div key={i}>
            {kv[1]}<span className="ds-k">{kv[2]}</span>:{kv[3] != null && ' '}
            {value && <span className={isString ? 'ds-s' : 'ds-v'}>{value}</span>}
          </div>
        )
      })}
    </>
  )
}
