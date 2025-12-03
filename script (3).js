// ===== THREE.JS ХВИЛЯ (покращена) =====

let scene, camera, renderer, points, clock;
let numX, numY, sep;
let flagMode = 0;
let transition = 0;
let transitionSpeed = 0.002;
let animateHeroRaf = null;
let prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const baseSep = 6;

function chooseGridSizes() {
  const w = Math.max(window.innerWidth, 320);
  if (w < 480) { numX = 120; numY = 60; sep = Math.round(baseSep * 1.2); }
  else if (w < 900) { numX = 220; numY = 100; sep = baseSep; }
  else if (w < 1400) { numX = 320; numY = 150; sep = baseSep; }
  else { numX = 420; numY = 200; sep = baseSep; }
}

function isWebGLAvailable() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
  } catch (e) {
    return false;
  }
}

function initHero() {
  chooseGridSizes();

  if (!isWebGLAvailable()) {
    document.body.style.background = 'linear-gradient(180deg,#5138ff 0%, #7f67ff 60%)';
    return;
  }

  // очищаємо старі ресурси при повторній ініціалізації
  if (renderer) {
    try {
      renderer.dispose?.();
      if (points) {
        scene.remove(points);
        points.geometry.dispose();
        points.material.dispose();
        points = null;
      }
    } catch (e) { /* ignore */ }
  }

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
  camera.position.set(0, 160, 180);

  const canvasEl = document.getElementById('wave');
  renderer = new THREE.WebGLRenderer({
    canvas: canvasEl,
    antialias: true,
    alpha: false
  });

  const maxDPR = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(maxDPR);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x5138ff, 1);

  const geo = new THREE.BufferGeometry();
  const total = numX * numY;
  const posArr = new Float32Array(total * 3);
  const colArr = new Float32Array(total * 3);
  let k = 0;
  for (let y = 0; y < numY; y++) {
    for (let x = 0; x < numX; x++) {
      posArr[k * 3] = (x - numX / 2) * sep;
      posArr[k * 3 + 1] = 0;
      posArr[k * 3 + 2] = (y - numY / 2) * sep;
      // початкові кольори (блакитна основа)
      colArr[k * 3] = 0;
      colArr[k * 3 + 1] = 0.34;
      colArr[k * 3 + 2] = 1;
      k++;
    }
  }
  geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));

  const mat = new THREE.PointsMaterial({ vertexColors: true, size: Math.max(1.0, Math.min(2.0, window.innerWidth / 900)) });
  points = new THREE.Points(geo, mat);
  scene.add(points);
  clock = new THREE.Clock();

  if (prefersReducedMotion) transitionSpeed = 0.0006;

  // start animation only if page is visible
  if (document.visibilityState === 'visible') startHeroAnimation();
}

function applyFlagColors() {
  if (!points) return;
  const colors = points.geometry.attributes.color;
  const n = colors.count;
  const midY = Math.floor(numY / 2);
  const midX = Math.floor(numX / 2);

  for (let i = 0; i < n; i++) {
    const y = Math.floor(i / numX);
    const x = i % numX;

    let r = 0, g = 0.34, b = 1; // default blue
    if (flagMode === 0) {
      if (y >= midY) { r = 1; g = 0.84; b = 0; }
    } else if (flagMode === 1) {
      const cross = Math.max(6, Math.floor(Math.min(numX, numY) * 0.03));
      const inVert = Math.abs(x - midX) < cross;
      const inHorz = Math.abs(y - midY) < cross;
      if (inVert || inHorz) { r = g = b = 1; } else { r = 0.8; g = 0.05; b = 0.15; }
    }

    const prevR = colors.getX(i), prevG = colors.getY(i), prevB = colors.getZ(i);
    colors.setXYZ(i,
      THREE.MathUtils.lerp(prevR, r, transition),
      THREE.MathUtils.lerp(prevG, g, transition),
      THREE.MathUtils.lerp(prevB, b, transition)
    );
  }
  colors.needsUpdate = true;
}

function heroStep() {
  if (!points || !clock) return;
  const t = clock.getElapsedTime() * (prefersReducedMotion ? 0.35 : 1.0);
  const pos = points.geometry.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    const d = Math.sqrt(x * x + z * z);
    pos.setY(i, Math.sin(d * 0.13 - t * 1.6) * (prefersReducedMotion ? 1.2 : 3));
  }
  pos.needsUpdate = true;

  transition += transitionSpeed;
  if (transition >= 1) {
    transition = 0;
    flagMode = (flagMode + 1) % 2;
    applyFlagColors();
  } else {
    applyFlagColors();
  }

  const camT = (clock.getElapsedTime() * 0.05);
  camera.position.x = Math.sin(camT) * 30;
  camera.lookAt(0, 0, 0);

  renderer.render(scene, camera);
}

function animateHero() {
  heroStep();
  animateHeroRaf = requestAnimationFrame(animateHero);
}

function startHeroAnimation() {
  if (prefersReducedMotion) return; // зберігаємо повагу до вибору користувача
  if (animateHeroRaf) cancelAnimationFrame(animateHeroRaf);
  animateHeroRaf = requestAnimationFrame(animateHero);
}

function stopHeroAnimation() {
  if (animateHeroRaf) cancelAnimationFrame(animateHeroRaf);
  animateHeroRaf = null;
}

let resizeTimeout = null;
function handleResize() {
  if (resizeTimeout) clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    if (!renderer || !camera) return;
    const prevNumX = numX;
    chooseGridSizes();
    if (prevNumX !== numX) { if (points) { scene.remove(points); points.geometry.dispose(); } initHero(); return; }
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    const maxDPR = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(maxDPR);
    renderer.setSize(window.innerWidth, window.innerHeight);
  }, 120);
}
window.addEventListener('resize', handleResize);

initHero();
window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener?.('change', (e) => {
  prefersReducedMotion = e.matches;
  transitionSpeed = prefersReducedMotion ? 0.0006 : 0.002;
});

// реагування на видимість вкладки
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    stopHeroAnimation();
  } else if (document.visibilityState === 'visible') {
    if (!prefersReducedMotion && isWebGLAvailable()) startHeroAnimation();
  }
});


// ===== TEAM SLIDER (робочий, доступний) =====
document.addEventListener('DOMContentLoaded', () => {
  // Елементи каруселі
  const btnPrev = document.querySelector('.team-button.prev');
  const btnNext = document.querySelector('.team-button.next');
  const teamList = document.querySelector('.team-list');
  const teamWindow = document.querySelector('.team-window');
  const statusEl = document.getElementById('team-status');
  const members = Array.from(document.querySelectorAll('.team-member'));

  if (!btnPrev || !btnNext || !teamList || !teamWindow || !members.length) return;

  let currentIndex = 0;

  function getGapValue() {
    const styles = window.getComputedStyle(teamList);
    const gap = parseFloat(styles.gap || styles.columnGap || styles.getPropertyValue('gap')) || 0;
    return Number.isFinite(gap) ? gap : 0;
  }

  function getMemberWidth() {
    const first = members[0];
    if (!first) return 0;
    return Math.round(first.getBoundingClientRect().width + getGapValue());
  }

  function getSliderMetrics() {
    const memberWidth = Math.max(1, getMemberWidth());
    const windowWidth = teamWindow.getBoundingClientRect().width || memberWidth;
    const visibleCount = Math.max(1, Math.floor(windowWidth / memberWidth));
    const maxIndex = Math.max(0, members.length - visibleCount);
    return { memberWidth, visibleCount, maxIndex, windowWidth };
  }

  function announceActiveMember() {
    if (!statusEl) return;
    const member = members[currentIndex];
    const name = member?.querySelector('h3')?.textContent?.trim();
    statusEl.textContent = name
      ? `Показано ${currentIndex + 1} з ${members.length}: ${name}`
      : `Показано ${currentIndex + 1} з ${members.length}`;
  }

  function updateButtons(maxIndex) {
    const atStart = currentIndex <= 0;
    const atEnd = currentIndex >= maxIndex;
    btnPrev.disabled = atStart;
    btnNext.disabled = atEnd;
    btnPrev.setAttribute('aria-disabled', atStart ? 'true' : 'false');
    btnNext.setAttribute('aria-disabled', atEnd ? 'true' : 'false');
  }

  function updateSlider({ announce = false } = {}) {
    teamWindow.classList.remove('team-window--cropped');
    teamWindow.style.removeProperty('--team-visible-width');

    const metrics = getSliderMetrics();
    const gap = getGapValue();
    const visibleWidth = Math.max(0, metrics.visibleCount * metrics.memberWidth - gap);
    const needsCrop = members.length > metrics.visibleCount;
    if (needsCrop) {
      teamWindow.classList.add('team-window--cropped');
      teamWindow.style.setProperty('--team-visible-width', `${visibleWidth}px`);
    } else {
      teamWindow.classList.remove('team-window--cropped');
      teamWindow.style.removeProperty('--team-visible-width');
    }
    currentIndex = Math.min(Math.max(0, currentIndex), metrics.maxIndex);
    const offset = -currentIndex * metrics.memberWidth;
    teamList.style.transform = `translateX(${offset}px)`;
    updateButtons(metrics.maxIndex);
    if (announce) announceActiveMember();
  }

  function attachButtonClickKey(button, handler) {
    button.addEventListener('click', handler);
    button.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handler();
      }
    });
  }

  attachButtonClickKey(btnPrev, () => {
    if (currentIndex <= 0) return;
    currentIndex -= 1;
    updateSlider({ announce: true });
    teamWindow.focus();
  });

  attachButtonClickKey(btnNext, () => {
    const { maxIndex } = getSliderMetrics();
    if (currentIndex >= maxIndex) return;
    currentIndex += 1;
    updateSlider({ announce: true });
    teamWindow.focus();
  });

  teamWindow.addEventListener('keydown', (e) => {
    const key = e.key;
    if (key === 'ArrowRight') {
      if (!btnNext.disabled) btnNext.click();
      e.preventDefault();
    } else if (key === 'ArrowLeft') {
      if (!btnPrev.disabled) btnPrev.click();
      e.preventDefault();
    } else if (key === 'Home') {
      currentIndex = 0;
      updateSlider({ announce: true });
      e.preventDefault();
    } else if (key === 'End') {
      currentIndex = getSliderMetrics().maxIndex;
      updateSlider({ announce: true });
      e.preventDefault();
    } else if (key === 'PageDown') {
      const metrics = getSliderMetrics();
      currentIndex = Math.min(metrics.maxIndex, currentIndex + metrics.visibleCount);
      updateSlider({ announce: true });
      e.preventDefault();
    } else if (key === 'PageUp') {
      const metrics = getSliderMetrics();
      currentIndex = Math.max(0, currentIndex - metrics.visibleCount);
      updateSlider({ announce: true });
      e.preventDefault();
    }
  });

  members.forEach((m, idx) => {
    m.setAttribute('tabindex', '0');
    m.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        currentIndex = Math.min(getSliderMetrics().maxIndex, idx);
        updateSlider({ announce: true });
        m.focus();
      }
    });
  });

  window.addEventListener('resize', () => {
    clearTimeout(window.__teamResizeTimeout);
    window.__teamResizeTimeout = setTimeout(() => updateSlider(), 120);
  });

  // drag / swipe
  let isDown = false, startX = 0;
  teamList.addEventListener('pointerdown', (e) => {
    isDown = true;
    startX = e.clientX;
    teamList.style.transition = 'none';
    teamList.setPointerCapture?.(e.pointerId);
  });
  teamList.addEventListener('pointermove', (e) => {
    if (!isDown) return;
    const dx = e.clientX - startX;
    teamList.style.transform = `translateX(${ -currentIndex * getMemberWidth() + dx }px)`;
  });
  teamList.addEventListener('pointerup', (e) => {
    if (!isDown) return;
    isDown = false;
    teamList.style.transition = '';
    const dx = e.clientX - startX;
    const threshold = getMemberWidth() * 0.25;
    if (dx < -threshold) {
      const { maxIndex } = getSliderMetrics();
      currentIndex = Math.min(maxIndex, currentIndex + 1);
    } else if (dx > threshold) {
      currentIndex = Math.max(0, currentIndex - 1);
    }
    updateSlider({ announce: true });
  });
  teamList.addEventListener('pointercancel', () => {
    isDown = false;
    teamList.style.transition = '';
    updateSlider();
  });

  updateSlider({ announce: true });

  /* ------------------------------
     NAV HIGHLIGHT (active on scroll)
     ------------------------------ */

  const navLinks = Array.from(document.querySelectorAll('.site-nav .nav-link[href^="#"]'));
  const sections = navLinks.map(a => {
    const id = a.getAttribute('href').slice(1);
    return document.getElementById(id);
  });

  let sectionObserver = null;
  function createSectionObserver() {
    // clean up
    if (sectionObserver) {
      sectionObserver.disconnect();
      sectionObserver = null;
    }

    const headerEl = document.querySelector('.site-header__inner');
    const headerHeight = headerEl ? Math.round(headerEl.getBoundingClientRect().height + 12) : 88;
    // rootMargin: top negative value to compensate header height, bottom to prefer earlier activation
    const rootMargin = `-${Math.round(headerHeight + 8)}px 0px -40% 0px`;

    sectionObserver = new IntersectionObserver((entries) => {
      // pick the most visible section (largest intersectionRatio)
      let visible = entries.filter(e => e.isIntersecting);
      if (visible.length === 0) {
        // if none intersecting, we still might want to mark based on bounding rect (topmost)
        entries.sort((a,b) => b.intersectionRatio - a.intersectionRatio);
      } else {
        visible.sort((a,b) => b.intersectionRatio - a.intersectionRatio);
      }
      const entry = visible[0] || entries[0];
      if (entry) {
        const id = entry.target.id;
        navLinks.forEach(a => {
          const href = a.getAttribute('href').slice(1);
          const isActive = href === id;
          a.classList.toggle('is-active', isActive && a.classList.contains('nav-button'));
          if (isActive) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
        });
      }
    }, { root: null, rootMargin, threshold: [0, 0.15, 0.35, 0.6] });

    sections.forEach(sec => {
      if (sec) sectionObserver.observe(sec);
    });
  }

  // init observer and update CSS var for header offset
  function updateHeaderOffsetAndObserver() {
    const headerEl = document.querySelector('.site-header__inner');
    const headerHeight = headerEl ? Math.round(headerEl.getBoundingClientRect().height + 12) : 88;
    document.documentElement.style.setProperty('--header-offset', `${headerHeight}px`);
    // також ставимо padding-top на body — щоб контент не перебував під фіксованим хедером
    try {
      document.body.style.paddingTop = `${headerHeight}px`;
    } catch (err) { /* ignore */ }

    createSectionObserver();
  }

  updateHeaderOffsetAndObserver();
  window.addEventListener('resize', () => {
    clearTimeout(window.__navResizeTimeout);
    window.__navResizeTimeout = setTimeout(() => updateHeaderOffsetAndObserver(), 120);
  });
});



/* ---------------------------
  Додаткові скрипти для Header (smooth scroll + toggle)
  --------------------------- */
(function () {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  document.addEventListener('click', (e) => {
    const a = e.target.closest && e.target.closest('.nav-link');
    if (!a) return;
    const href = a.getAttribute('href') || '';
    if (!href.startsWith('#')) return;
    const targetId = href.slice(1);
    const target = document.getElementById(targetId);
    if (!target) return;
    e.preventDefault();

    const headerEl = document.querySelector('.site-header__inner');
    const headerHeight = headerEl ? headerEl.getBoundingClientRect().height + 12 : 88;

    const top = target.getBoundingClientRect().top + window.scrollY - headerHeight;

    if (prefersReducedMotion) {
      window.scrollTo(0, top);
      target.focus({ preventScroll: true });
    } else {
      window.scrollTo({ top, behavior: 'smooth' });
      setTimeout(() => {
        try { target.focus(); } catch (err) { /* silent */ }
      }, 420);
    }
  });

  const toggle = document.querySelector('.site-header__toggle');
  const header = document.querySelector('.site-header');
  if (toggle && header) {
    toggle.addEventListener('click', () => {
      const expanded = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', (!expanded).toString());
      header.setAttribute('aria-expanded', (!expanded).toString());
    });
  }
})();
