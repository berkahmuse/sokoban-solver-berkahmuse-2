// Room-decomposed push-optimal Sokoban solver.
//
// One program answers a whole set (5 instances) in a single run. Each level is
// a hallway with 1-4 rooms off it (3-4 boxes each); boxes never leave their
// room, so the level optimum is the sum of the rooms' optima. Every room is
// solved on its own with A* over pushes, then the room solutions are joined
// with the player's walks between them.
//
// A push-state is the boxes' squares plus the player's reachable region (named
// by its lowest square). The heuristic is the min-cost matching of boxes to
// goals over shortest-path floor distances, which never overestimates the
// pushes left, so the first solved state A* pops is push-optimal. Dead states
// are pruned two ways: a box on a dead square (one no push can ever bring to a
// goal) and a box frozen in a 2x2 block of walls and boxes. Work is budgeted
// by expansions, never by the clock, and every tie is broken by a fixed order,
// so two runs write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
/** Expansions one room's search may spend; the whole level's fallback may spend this many. */
const ROOM_EXPANSIONS = 150_000;
const GLOBAL_EXPANSIONS = 300_000;
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text: # wall, space floor, . goal, $ box, * box on goal, @ player. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares a box can be pushed from onto some goal, pulled back from the goals over open floor. */
function liveSquares(lv: Level): Uint8Array {
  const { w, wall, goals } = lv;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** The squares the player reaches from `from` with `occupied` in the way; returns the lowest one. */
function regionLow(lv: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = lv;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(lv: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = lv;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (u first on ties). */
function walkTo(lv: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = lv;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no goal) out of the
 * floor, and each part left that holds a box is a room. Null when some part holds more or fewer boxes than goals, or
 * a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(lv: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const live = liveSquares(lv);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]]!.boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]]!.goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** A state's key: the region's lowest square, then the boxes' squares, packed base 256 (exact under 2**53). */
function keyOf(boxes: number[], low: number): number {
  let k = low;
  for (const b of boxes) k = k * 256 + b;
  return k;
}

/**
 * The push-optimal solution of `level`, searching at most `maxExpansions` states: {pushes, moves} (moves in LURD, an
 * upper letter a push), or null when the search gives up (or the level has no solution). A* over pushes: the estimate
 * is the min-cost matching of boxes to goals over floor distances, which never overestimates, so the first solved
 * state popped is push-optimal.
 */
function solvePushOptimal(lv: Level, maxExpansions: number): { pushes: number; moves: string } | null {
  const { w, wall, goals } = lv;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(lv);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (lv.boxes.some((b) => !live[b])) return null;

  // Floor distances from each goal, ignoring boxes: dist[j][cell], -1 when unreachable.
  const dist: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi];
      for (const [dx, dy] of DIRS) {
        const n = p + dy * w + dx;
        if (!wall[n] && d[n] < 0) {
          d[n] = d[p] + 1;
          q.push(n);
        }
      }
    }
    return d;
  });
  // Min-cost matching of boxes to goals over those distances (n <= 4: try every assignment).
  const heuristic = (boxes: number[]): number => {
    const n = boxes.length;
    const perm = goals.map((_, i) => i);
    let best = Infinity;
    const tryPerm = (k: number): void => {
      if (k === n) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          const d = dist[perm[j]]![boxes[j]]!;
          if (d < 0) return;
          sum += d;
        }
        if (sum < best) best = sum;
        return;
      }
      for (let j = k; j < n; j++) {
        [perm[k], perm[j]] = [perm[j]!, perm[k]!];
        tryPerm(k + 1);
        [perm[k], perm[j]] = [perm[j]!, perm[k]!];
      }
    };
    tryPerm(0);
    return best;
  };

  // The states in the order found: their boxes, the player's square, and the push that made them from their parent.
  const statesBoxes: number[][] = [lv.boxes];
  const statesPlayer: number[] = [lv.player];
  const parent: number[] = [-1];
  const pushDir: number[] = [-1];
  const pushFrom: number[] = [-1];
  const best = new Map<number, number>();
  for (const b of lv.boxes) occupied[b] = 1;
  best.set(keyOf(lv.boxes, regionLow(lv, occupied, lv.player, seen, ++stamp, stack)), 0);
  for (const b of lv.boxes) occupied[b] = 0;
  const h0 = heuristic(lv.boxes);
  if (h0 === Infinity) return null;
  if (solved(lv.boxes)) return { pushes: 0, moves: '' };
  // Open list: a heap on (f, -g, sequence), so ties go to the deeper state, then the earlier one found.
  const heap: [number, number, number, number][] = [[h0, 0, 0, 0]];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  let seq = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, gneg, , s] = heapPop();
    const g = -gneg;
    const boxes = statesBoxes[s]!;
    for (const b of boxes) occupied[b] = 1;
    const low = regionLow(lv, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    if (best.get(keyOf(boxes, low)) !== g) {
      for (const b of boxes) occupied[b] = 0;
      continue;
    }
    if (solved(boxes)) {
      found = s;
      for (const b of boxes) occupied[b] = 0;
      break;
    }
    if (++expanded > maxExpansions) {
      for (const b of boxes) occupied[b] = 0;
      return null;
    }
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const [dx, dy] = DIRS[k]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(lv, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          h = heuristic(next);
          if (h !== Infinity) key = keyOf(next, regionLow(lv, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity) continue;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= g + 1) continue;
        best.set(key, g + 1);
        const idx = statesBoxes.length;
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(s);
        pushDir.push(k);
        pushFrom.push(b);
        heapPush([g + 1 + h, -(g + 1), ++seq, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = pushFrom[s]!;
    const k = pushDir[s]!;
    const d = DIRS[k]![1] * w + DIRS[k]![0];
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(lv, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[k]![3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** The level solved room by room: each room's push-optimal solution in turn, the other rooms' boxes standing as walls. */
function solveByRooms(lv: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(lv);
  if (rooms === null) return null;
  let player = lv.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = lv.wall.slice();
    for (const other of rooms) if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    const sub: Level = {
      w: lv.w,
      h: lv.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const r = solvePushOptimal(sub, ROOM_EXPANSIONS);
    if (r === null) return null;
    pushes += r.pushes;
    moves += r.moves;
    player = endOf(lv.w, player, r.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/** The moves for a level: room by room when it is a set of rooms, else one search over the whole level. */
function solve(text: string): string {
  const lv = parseLevel(text);
  const byRooms = solveByRooms(lv);
  if (byRooms !== null) return byRooms.moves;
  const whole = solvePushOptimal(lv, GLOBAL_EXPANSIONS);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
