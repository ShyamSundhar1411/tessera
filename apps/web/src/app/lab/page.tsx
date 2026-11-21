import type { Metadata } from "next";
import Lab from "@/components/lab/lab";

export const metadata: Metadata = {
  title: "Lab",
  description: "Load, compare, rewire, train and export neural network architectures in your browser.",
  alternates: { canonical: "/lab/" },
};

export default function LabPage() {
  return <Lab />;
}
