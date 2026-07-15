// onAction handler for interpreted mode. Maps OpenUI Lang actions (fired by
// Button(..., Action([...]))) to app behaviour: in-app navigation for internal
// paths, window.open for external URLs, and a console log of anything unhandled
// (useful signal for the parity loop / DSL correction).
type Nav = (to: string) => void;

function urlsIn(obj: any, out: string[] = []): string[] {
  if (obj == null) return out;
  if (typeof obj === "string") {
    if (/^(https?:\/\/|\/)/.test(obj)) out.push(obj);
    return out;
  }
  if (Array.isArray(obj)) {
    obj.forEach((v) => urlsIn(v, out));
    return out;
  }
  if (typeof obj === "object") {
    for (const v of Object.values(obj)) urlsIn(v, out);
  }
  return out;
}

export function makeOnAction(navigate: Nav) {
  return (event: any) => {
    // The @OpenUrl builtin carries a url; internal ("/x") -> client route,
    // external -> new tab. We scan the event defensively since the exact
    // ActionEvent shape varies by action composition.
    const urls = urlsIn(event);
    const internal = urls.find((u) => u.startsWith("/"));
    const external = urls.find((u) => /^https?:\/\//.test(u));
    if (internal) {
      navigate(internal);
      return;
    }
    if (external) {
      window.open(external, "_blank", "noopener");
      return;
    }
    // Mutations (@Run) are executed by the toolProvider; nothing to do here.
    // Log unrecognised actions so the parity loop can wire them.
    if (event && event.type !== "form_submit") {
      // eslint-disable-next-line no-console
      console.debug("[interpret] unhandled action", event);
    }
  };
}
