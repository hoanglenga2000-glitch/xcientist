export function decodeXmlText(value: string) {
  const cdata: string[] = [];
  const markerPrefix = "\uE000EVOMIND_CDATA_";
  const markerSuffix = "\uE001";
  const protectedText = value.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, (_match, content: string) => {
    const index = cdata.push(content) - 1;
    return `${markerPrefix}${index}${markerSuffix}`;
  });
  const decoded = protectedText
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&apos;", "'")
    .replaceAll("&amp;", "&");
  return decoded.replace(new RegExp(`${markerPrefix}(\\d+)${markerSuffix}`, "g"), (_match, rawIndex: string) => {
    const index = Number(rawIndex);
    return Number.isSafeInteger(index) && index >= 0 && index < cdata.length ? cdata[index] : "";
  });
}
