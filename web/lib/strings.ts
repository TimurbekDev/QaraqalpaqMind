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

  // --- Actions ---
  newChat: "Jańa sáwbet",
  send: "Jiberiw",
  stop: "Toqtatıw",
  continueGen: "Dawam etiw",
  regenerate: "Qaytadan juwap",
  retry: "Qaytadan urınıw",
  copy: "Kóshiriw",
  copied: "Kóshirildi",
  edit: "Ózgertiw",
  save: "Saqlaw",
  cancel: "Biykar etiw",
  delete: "Óshiriw",
  rename: "Atın ózgertiw",
  close: "Jabıw",
  scrollToLatest: "Eń jańasına ótiw",

  // --- Sidebar ---
  conversations: "Sáwbetler",
  searchPlaceholder: "Sáwbetlerden izlew",
  noResults: "Hesh nárse tabılmadı",
  noConversations: "Sáwbetler joq",
  untitled: "Atsız sáwbet",
  today: "Búgin",
  thisWeek: "Usı hápte",
  older: "Buracıraq",
  openSidebar: "Dizimdi ashıw",
  closeSidebar: "Dizimdi jabıw",
  deleteConfirm: "Bul sáwbet óshirilsin be?",

  // --- Empty state ---
  emptyTitle: "Sálem!",
  emptyBody: "Qaraqalpaq tilinde sorawıńızdı jazıń.",

  // --- Composer ---
  placeholder: "Xabarıńızdı jazıń...",
  disclaimer: "Model qátelesiwi múmkin. Áhmiyetli maǵlıwmatlardı tekseriń.",

  // --- Roles ---
  you: "Siz",
  assistant: "QaraqalpaqMind",

  // --- Status ---
  thinking: "Oylanıp atır...",
  stopped: "Toqtatıldı",
  errorGeneric: "Qátelik júz berdi. Qaytadan urınıp kóriń.",
  errorUnreachable: "Server menen baylanıs joq.",
  errorRateLimited: "Júdá kóp soraw. Biraz kútiń.",

  // --- Settings ---
  settings: "Sazlawlar",
  theme: "Kórinis",
  themeSystem: "Sistema",
  themeLight: "Jaqtı",
  themeDark: "Qarańǵı",
  temperature: "Temperatura",
  temperatureHint: "Tómen bolsa - anıq, joqarı bolsa - erkin juwap.",
  systemPrompt: "Sistema kórsetpesi",
  systemPromptHint: "Modelge hár sáwbette beriletuǵın baslanǵısh kórsetpe.",
  sendOnEnter: "Enter menen jiberiw",
  sendOnEnterHint: "Óshirilse, Enter jańa qatar qosadı.",
  dangerZone: "Qáwipli aymaq",
  clearAll: "Barlıq sáwbetlerdi óshiriw",
  clearAllConfirm: "Barlıq sáwbetler óshirilsin be? Bul qaytarılmaydı.",

  // --- Shortcuts ---
  shortcuts: "Klavishalar",
  shortcutNew: "Jańa sáwbet",
  shortcutSearch: "Izlew",
  shortcutSidebar: "Dizimdi ashıw/jabıw",
  shortcutSettings: "Sazlawlar",
  shortcutStop: "Juwaptı toqtatıw",
  shortcutFocus: "Jazıw maydanına ótiw",
  shortcutShortcuts: "Usı dizimdi kórsetiw",

  // --- Accessibility, not shown visually ---
  srGenerating: "Juwap jazılıp atır",
  srSkipToInput: "Jazıw maydanına ótiw",
  srMessageActions: "Xabar ámelleri",
} as const;

/** Suggested opening prompts, shown on the empty state. */
export const suggestions: readonly { title: string; prompt: string }[] = [
  { title: "Tanıstırıw", prompt: "Qaraqalpaqstan haqqında qısqasha aytıp ber" },
  { title: "Geografiya", prompt: "Ámiwdárya qayerden baslanıp, qayerge quyadı?" },
  { title: "Dóretiwshilik", prompt: "Qaraqalpaq tilinde qısqa gúrriń jaz" },
  { title: "Awdarma", prompt: "«Kitap oqıw paydalı» degen sózdi orıs tiline awdar" },
];
