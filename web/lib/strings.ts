/**
 * Every user-visible string, in Karakalpak (Latin 2016 orthography).
 *
 * Collected here rather than scattered through components so a native speaker
 * can review the whole interface without reading React. These were written by
 * a non-native speaker and are the weakest part of this app - treat them as a
 * first draft awaiting correction, the same way the SFT seed examples are.
 *
 * Orthography note: 2016 Latin uses the acute letters á ó ú ǵ ń and dotless ı.
 * Do not "fix" ı to i or á to a - they are distinct letters.
 */
export const kaa = {
  appName: "QaraqalpaqMind",
  tagline: "Qaraqalpaq tilindegi jasalma intellekt",

  newChat: "Jańa sáwbet",
  placeholder: "Xabarıńızdı jazıń...",
  send: "Jiberiw",
  stop: "Toqtatıw",

  emptyTitle: "Sálem!",
  emptyBody: "Qaraqalpaq tilinde sorawıńızdı jazıń.",

  you: "Siz",
  assistant: "QaraqalpaqMind",

  thinking: "Oylanıp atır...",
  errorGeneric: "Qátelik júz berdi. Qaytadan urınıp kóriń.",
  errorUnreachable: "Server menen baylanıs joq.",
  errorRateLimited: "Júdá kóp soraw. Biraz kútiń.",

  /** Shown once, under the composer. */
  disclaimer: "Model qátelesiwi múmkin. Áhmiyetli maǵlıwmatlardı tekseriń.",
} as const;

/** Suggested opening prompts, shown on the empty state. */
export const suggestions: readonly string[] = [
  "Qaraqalpaqstan haqqında aytıp ber",
  "Ámiwdárya qayerden baslanadı?",
  "Qaraqalpaq tilinde qısqa gúrriń jaz",
];
