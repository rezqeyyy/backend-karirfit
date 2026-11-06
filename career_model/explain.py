"""Deterministic template explanation – shown when the LLM is unavailable (NFR6) and used as
the factual skeleton. Mirrors templateExplain() in web/app.js."""


def template_explanation(result: dict, names: dict, lang: str = "id") -> str:
    n = lambda r: names.get(r["skill"], r["skill"])
    matched = sorted(result["matched"], key=lambda r: -r["weight"])
    gaps = result["gaps"]
    career = result["name_id"] if lang == "id" else result["name_en"]
    if lang == "id":
        parts = [
            f"1. Dasar kecocokan\nSkor kecocokan Anda untuk {career} adalah {result['fit']:.0f}/100: "
            f"kecocokan kompetensi {result['K']:.0%}"
            + (" dan minat Anda termasuk bidang karier ini." if result["M"] else "; minat yang Anda pilih tidak termasuk bidang ini."),
            "2. Kompetensi yang sudah sesuai\n" + ("\n".join(
                f"- {n(r)} (diminta di {r['demand']:.0%} lowongan)" for r in matched[:6]) or "- Belum ada keterampilan yang cocok."),
            "3. Kompetensi yang perlu ditingkatkan\n" + ("\n".join(
                f"- {n(r)} (diminta di {r['demand']:.0%} lowongan)" for r in gaps[:5]) or "- Tidak ada kesenjangan utama."),
            "4. Saran pengembangan\n" + (
                f"Mulailah dari {', '.join(n(r) for r in gaps[:3])}: keterampilan ini paling berbobot untuk {career}. "
                "Buat satu proyek portofolio kecil yang memakainya, lalu tambahkan ke pengalaman Anda."
                if gaps else "Perdalam keterampilan yang ada lewat proyek nyata dan sertifikasi."),
        ]
    else:
        parts = [
            f"1. Basis of fit\nYour fit score for {career} is {result['fit']:.0f}/100: competency fit {result['K']:.0%}"
            + (" and your interests include this career area." if result["M"] else "; your selected interests do not include this area."),
            "2. Matched competencies\n" + ("\n".join(
                f"- {n(r)} (requested in {r['demand']:.0%} of job ads)" for r in matched[:6]) or "- No matching skills yet."),
            "3. Competencies to improve\n" + ("\n".join(
                f"- {n(r)} (requested in {r['demand']:.0%} of job ads)" for r in gaps[:5]) or "- No major gaps."),
            "4. Development advice\n" + (
                f"Start with {', '.join(n(r) for r in gaps[:3])}: these carry the most weight for {career}. "
                "Build one small portfolio project that uses them and add it to your experience."
                if gaps else "Deepen your current skills through real projects and certifications."),
        ]
    return "\n\n".join(parts)
