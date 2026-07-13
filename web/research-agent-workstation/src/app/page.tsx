import HomeClient from "./home-client";

type HomeProps = {
  searchParams: Promise<{
    page?: string | string[];
  }>;
};

export default async function Home({ searchParams }: HomeProps) {
  const { page } = await searchParams;
  const initialPage = Array.isArray(page) ? page[0] : page;

  return <HomeClient initialPage={initialPage} />;
}
