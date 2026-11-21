// Vite imports a file's text with the ?raw suffix (used to read wrangler.jsonc in tests).
declare module '*?raw' {
  const text: string;
  export default text;
}
