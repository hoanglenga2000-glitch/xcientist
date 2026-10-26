"use client";

import { useEffect } from "react";
import { workspaceLocation } from "@/components/workstation/task-workspace/workspace-location";

type HomeClientProps = {
  initialPage?: string;
};

/** Classic shell is not an entry. Old page ids land in the workspace. */
export default function HomeClient({ initialPage }: HomeClientProps) {
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (initialPage && !params.get("page")) params.set("page", initialPage);
    window.location.replace(workspaceLocation(params));
  }, [initialPage]);
  return null;
}
