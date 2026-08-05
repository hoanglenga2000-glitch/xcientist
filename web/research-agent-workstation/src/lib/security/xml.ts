export function decodeXmlText(value: string) {
  const cdata: string[] = [];
  const protectedText = value.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, (_match, content: string) => {
    const index = cdata.push(content) - 1;
    return `\u0000EVOMIND_CDATA_${index}\u0000`;
  });
  const decoded = protectedText
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&apos;", "'")
    .replaceAll("&amp;", "&");
  return decoded.replace(/\u0000EVOMIND_CDATA_(\d+)\u0000/g, (_match, rawIndex: string) => {
    const index = Number(rawIndex);
    return Number.isSafeInteger(index) && index >= 0 && index < cdata.length ? cdata[index] : "";
  });
}
