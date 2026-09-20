export function StreamingDots() {
  return (
    <div className="flex items-center gap-1 p-3">
      <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400" />
    </div>
  )
}