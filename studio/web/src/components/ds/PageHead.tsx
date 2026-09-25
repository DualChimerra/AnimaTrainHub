/** Page header from the mockup (.pagehead): mono eyebrow, 25px title,
 *  subtitle, tools on the right. */
export default function PageHead({ eyebrow, accent = false, title, subtitle, tools }: {
  eyebrow?: React.ReactNode
  /** Green eyebrow, used on version steps ("ШАГ 3 · ВЕРСИЯ v3-lokr"). */
  accent?: boolean
  title: React.ReactNode
  subtitle?: React.ReactNode
  tools?: React.ReactNode
}) {
  return (
    <div className="ds-pagehead">
      <div className="ds-pagehead-txt">
        {eyebrow && <div className={`ds-eyebrow${accent ? ' ds-accent' : ''}`}>{eyebrow}</div>}
        <h1>{title}</h1>
        {subtitle && <div className="ds-subtitle">{subtitle}</div>}
      </div>
      {tools && <div className="ds-headtools">{tools}</div>}
    </div>
  )
}
