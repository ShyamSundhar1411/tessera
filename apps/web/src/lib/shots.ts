import { BASE_PATH } from "./site";

/** A screenshot of the lab in public/shots, captured from the running app. */
export const shotUrl = (id: string) => `${BASE_PATH}/shots/${id}.webp`;
