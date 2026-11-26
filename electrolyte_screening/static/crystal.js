const colors = {
  Li: "#8edcca", Na: "#beafff", K: "#bc94f6", Mg: "#86e1aa", Al: "#bfc9e8",
  Si: "#f3c681", P: "#f6a464", S: "#f2d16e", O: "#f58da6", F: "#abda76",
  Cl: "#9bd66c", Br: "#e9957c", I: "#ae9cf7", Ge: "#9cbbf8", N: "#8cb8f5",
};
const fallback = ["#83c9ef", "#e8abf3", "#f3bd8a", "#a9bbfa", "#71d3bc", "#e8a8a9"];
const corners = Array.from({length: 8}, (_, n) => [n & 1, n >> 1 & 1, n >> 2 & 1]);
const edges = corners.flatMap((_, n) => [1, 2, 4].filter(bit => !(n & bit)).map(bit => [n, n | bit]));

export function elementColor(element) {
  return colors[element] ?? fallback[[...element].reduce((value, letter) => value * 11 + letter.charCodeAt(0), 0) % fallback.length];
}

function centerInCell(fractional, matrix) {
  return matrix[0].map((_, axis) => matrix.reduce(
    (total, vector, i) => total + vector[axis] * (fractional[i] - 0.5), 0,
  ));
}

export function crystalViewer(canvas, structure, onSelection = null) {
  const context = canvas.getContext("2d");
  const referenceCorners = corners.map(point => centerInCell(point, structure.cell));
  const extent = Math.max(...referenceCorners.map(point => Math.hypot(...point))) || 1;
  const sites = structure.sites.map(site => {
    const point = site.fractional.map(value => value - Math.floor(value + 1e-9));
    const species = Object.entries(site.species).sort((a, b) => b[1] - a[1]);
    return {point: centerInCell(point, structure.cell), site, color: elementColor(species[0][0])};
  });
  const interactive = onSelection !== null;
  const controller = new AbortController();
  const {signal} = controller;
  let pitch = -0.38;
  let yaw = 0.62;
  let zoom = 1;
  let showCell = true;
  let selected = -1;
  let markers = [];
  let drag = null;

  function draw() {
    const {width, height} = canvas.getBoundingClientRect();
    if (!context || !width || !height) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelsWide = Math.round(width * pixelRatio);
    const pixelsHigh = Math.round(height * pixelRatio);
    if (canvas.width !== pixelsWide || canvas.height !== pixelsHigh) {
      canvas.width = pixelsWide;
      canvas.height = pixelsHigh;
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    const unit = Math.min(width, height) * (interactive ? 0.38 : 0.41) * zoom;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch);
    const centerX = width * (interactive ? 0.5 : 0.51);
    const centerY = height * (interactive ? 0.53 : 0.48);
    function project([x, y, z]) {
      const xx = (x * cy + z * sy) / extent;
      const yz = (-x * sy + z * cy) / extent;
      const yy = y / extent * cx - yz * sx;
      const depth = y / extent * sx + yz * cx;
      const perspective = 3.5 / (3.5 - depth);
      return {x: centerX + xx * perspective * unit, y: centerY - yy * perspective * unit, depth, perspective};
    }
    if (showCell) {
      const cell = referenceCorners.map(project);
      context.lineWidth = interactive ? 1.35 : 1.5;
      for (const [i, j] of edges) {
        context.strokeStyle = cell[i].depth + cell[j].depth > 0 ? "#c8b5f5aa" : "#a18ac76d";
        context.setLineDash(cell[i].depth + cell[j].depth > 0 ? [] : [4, 4]);
        context.beginPath();
        context.moveTo(cell[i].x, cell[i].y);
        context.lineTo(cell[j].x, cell[j].y);
        context.stroke();
      }
      context.setLineDash([]);
    }
    const radius = interactive ? Math.max(3.3, Math.min(15, unit * 0.135 / Math.cbrt(sites.length))) : unit * 0.116;
    markers = sites.map((site, i) => ({...project(site.point), color: site.color, i}))
      .sort((a, b) => a.depth - b.depth);
    for (const marker of markers) {
      const {x, y, color, perspective, i} = marker;
      marker.radius = radius * perspective;
      const r = marker.radius;
      const gradient = context.createRadialGradient(x - r * 0.28, y - r * 0.34, r * 0.04, x, y, r);
      gradient.addColorStop(0, "#ffffff");
      gradient.addColorStop(0.36, color);
      gradient.addColorStop(0.85, color);
      gradient.addColorStop(1, "#262333");
      if (!interactive) {
        context.shadowColor = color;
        context.shadowBlur = 11;
      }
      context.fillStyle = gradient;
      context.beginPath();
      context.arc(x, y, r, 0, Math.PI * 2);
      context.fill();
      context.shadowBlur = 0;
      if (i === selected) {
        context.strokeStyle = "#fff";
        context.lineWidth = 1.5;
        context.beginPath();
        context.arc(x, y, r + 3, 0, Math.PI * 2);
        context.stroke();
      }
    }
  }

  function select(index) {
    selected = index;
    onSelection?.(index >= 0 ? sites[index].site : null, index);
    draw();
  }

  if (interactive) {
    canvas.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      drag = {x: event.clientX, y: event.clientY, moved: false};
      canvas.setPointerCapture(event.pointerId);
    }, {signal});
    canvas.addEventListener("pointermove", event => {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      drag.moved ||= Math.abs(dx) + Math.abs(dy) > 2;
      yaw += dx * 0.012;
      pitch += dy * 0.012;
      [drag.x, drag.y] = [event.clientX, event.clientY];
      draw();
    }, {signal});
    canvas.addEventListener("pointerup", event => {
      if (drag && !drag.moved) {
        const rect = canvas.getBoundingClientRect();
        const hit = markers.findLast(marker => Math.hypot(
          marker.x - event.clientX + rect.left, marker.y - event.clientY + rect.top,
        ) <= marker.radius + 4);
        select(hit?.i ?? -1);
      }
      drag = null;
    }, {signal});
    canvas.addEventListener("pointercancel", () => { drag = null; }, {signal});
    canvas.addEventListener("wheel", event => {
      event.preventDefault();
      zoom = Math.max(0.48, Math.min(2.8, zoom * Math.exp(-event.deltaY * 0.001)));
      draw();
    }, {signal, passive: false});
    canvas.addEventListener("keydown", event => {
      switch (event.key) {
        case "ArrowLeft": yaw -= 0.14; break;
        case "ArrowRight": yaw += 0.14; break;
        case "ArrowUp": pitch -= 0.14; break;
        case "ArrowDown": pitch += 0.14; break;
        case "+": case "=": zoom = Math.min(2.8, zoom * 1.15); break;
        case "-": zoom = Math.max(0.48, zoom / 1.15); break;
        case "Enter": select((selected + 1) % sites.length); break;
        case "Escape": select(-1); break;
        default: return;
      }
      event.preventDefault();
      draw();
    }, {signal});
  }
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  draw();
  return {
    reset() {
      pitch = -0.38;
      yaw = 0.62;
      zoom = 1;
      select(-1);
    },
    setCell(visible) { showCell = visible; draw(); },
    destroy() { controller.abort(); observer.disconnect(); },
  };
}

export function drawIllustration(canvas) {
  const sites = [
    ...corners.map(fractional => ({fractional, species: {I: 1}})),
    ...[0, 1, 2].flatMap(axis => [0, 0.99999].map(edge => ({
      fractional: Array.from({length: 3}, (_, i) => i === axis ? edge : 0.5),
      species: {Li: 1},
    }))),
    {fractional: [0.5, 0.5, 0.5], species: {I: 1}},
  ];
  for (const site of sites) site.fractional = site.fractional.map(value => value === 1 ? 0.99999 : value);
  crystalViewer(canvas, {cell: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], sites});
}
