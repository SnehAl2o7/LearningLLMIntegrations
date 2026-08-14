export default function LoadingSkeleton({ count = 16 }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className={`rounded-xl overflow-hidden stagger-${Math.min(i + 1, 16)}`}>
          {/* Cover skeleton */}
          <div className="aspect-[2/3] skeleton" />
          {/* Title skeleton */}
          <div className="p-4 space-y-2.5">
            <div className="h-4 w-3/4 skeleton rounded" />
            <div className="h-3 w-1/2 skeleton rounded" />
            <div className="flex items-center gap-2 mt-1">
              <div className="h-3 w-16 skeleton rounded-full" />
              <div className="h-3 w-20 skeleton rounded-full" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
