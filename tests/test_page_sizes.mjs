import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../pages/dashboard/app.js', import.meta.url), 'utf8')
  .replace(/^init\(\)\.catch.*$/m, '');

function loadUi() {
  const nodes = new Map();
  for (const id of [
    'generationForm', 'modelSelect', 'aspectRatioField', 'resolutionField',
    'aspectRatioSelect', 'resolutionSelect', 'imageWidthField', 'imageHeightField',
    'imageSizeHint', 'imageWidthInput', 'imageHeightInput', 'promptInput',
    'imageCountInput', 'submitBtn',
  ]) {
    let value = '';
    const node = {
      hidden: false,
      disabled: false,
      required: false,
      listeners: {},
      addEventListener(name, callback) { this.listeners[name] = callback; },
      appendChild(option) {
        if (option.selected || !value) value = String(option.value);
      },
    };
    Object.defineProperty(node, 'value', {
      get: () => value,
      set: (next) => { value = String(next); },
    });
    Object.defineProperty(node, 'innerHTML', {
      set: () => { value = ''; },
    });
    nodes.set(`#${id}`, node);
  }
  const calls = [];
  const messages = [];
  const context = vm.createContext({
    document: {
      querySelector: (selector) => nodes.get(selector) || null,
      querySelectorAll: () => [],
      addEventListener() {},
      createElement: () => ({}),
    },
    window: { addEventListener() {} },
    calls,
    messages,
  });
  vm.runInContext(source, context);
  vm.runInContext(`
    appState.pluginState = {
      current_model: 'Custom name/portrait',
      available_models: ['Custom name/portrait', 'NovelAI/chat', 'Other NovelAI/square'],
      model_settings: {
        'Custom name/portrait': {adapter_type: 'novelai', width: 896, height: 1152},
        'NovelAI/chat': {adapter_type: 'openai_chat'},
        'Other NovelAI/square': {adapter_type: 'novelai', width: 512, height: 512},
      },
    };
    apiPost = async (endpoint, payload) => {
      calls.push({endpoint, payload: JSON.parse(JSON.stringify(payload))});
      return {};
    };
    showToast = (message, isError) => messages.push({message, isError});
    renderGenerateResults = () => {};
    loadTasks = async () => {};
    loadStrings = async () => {};
    applyStaticText = () => {};
    renderTaskList = () => {};
    renderOverviewSummary = () => {};
    bindEvents();
  `, context);
  nodes.get('#modelSelect').value = 'Custom name/portrait';
  nodes.get('#promptInput').value = 'cat ears';
  nodes.get('#imageCountInput').value = '1';
  nodes.get('#aspectRatioSelect').value = '3:4';
  nodes.get('#resolutionSelect').value = '2K';
  return { context, nodes, calls, messages };
}

test('model changes use adapter metadata and show the configured NovelAI pixel size', () => {
  const { nodes } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#aspectRatioField').hidden, true);
  assert.equal(nodes.get('#resolutionSelect').disabled, true);
  assert.equal(nodes.get('#imageWidthField').hidden, false);
  assert.equal(nodes.get('#imageWidthInput').required, true);
  assert.equal(nodes.get('#imageWidthInput').value, '896');
  assert.equal(nodes.get('#imageHeightInput').value, '1152');

  nodes.get('#modelSelect').value = 'NovelAI/chat';
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#aspectRatioField').hidden, false);
  assert.equal(nodes.get('#resolutionSelect').disabled, false);
  assert.equal(nodes.get('#imageWidthField').hidden, true);
  assert.equal(nodes.get('#imageWidthInput').disabled, true);
  assert.equal(nodes.get('#imageWidthInput').required, false);

  nodes.get('#modelSelect').value = 'Other NovelAI/square';
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#imageWidthInput').value, '512');
  assert.equal(nodes.get('#imageHeightInput').value, '512');
});

test('NovelAI submits explicit pixel sizes without competing aspect-ratio fields', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  nodes.get('#imageWidthInput').value = '1152';
  nodes.get('#imageHeightInput').value = '896';
  await vm.runInContext('submitTask({preventDefault() {}})', context);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].endpoint, 'page/generate');
  assert.equal(calls[0].payload.width, 1152);
  assert.equal(calls[0].payload.height, 896);
  assert.equal(calls[0].payload.prompt, 'cat ears');
  assert.equal(Object.hasOwn(calls[0].payload, 'aspect_ratio'), false);
  assert.equal(Object.hasOwn(calls[0].payload, 'resolution'), false);
  assert.equal(nodes.get('#submitBtn').disabled, false);
});

test('state refresh and locale changes preserve an unsubmitted model and manual pixel size', async () => {
  const { context, nodes } = loadUi();
  vm.runInContext('renderState()', context);
  nodes.get('#modelSelect').value = 'Other NovelAI/square';
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#imageWidthInput').value, '512');
  nodes.get('#imageWidthInput').value = '1024';
  nodes.get('#imageHeightInput').value = '768';

  vm.runInContext('renderState()', context);
  assert.equal(nodes.get('#modelSelect').value, 'Other NovelAI/square');
  assert.equal(nodes.get('#imageWidthInput').value, '1024');
  assert.equal(nodes.get('#imageHeightInput').value, '768');

  await vm.runInContext('refreshLocalizedUi()', context);
  assert.equal(nodes.get('#modelSelect').value, 'Other NovelAI/square');
  assert.equal(nodes.get('#imageWidthInput').value, '1024');
  assert.equal(nodes.get('#imageHeightInput').value, '768');

  nodes.get('#imageHeightInput').value = '';
  nodes.get('#modelSelect').listeners.change();
  vm.runInContext('renderState()', context);
  assert.equal(nodes.get('#imageHeightInput').value, '');

  nodes.get('#modelSelect').value = 'Custom name/portrait';
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#imageWidthInput').value, '896');
  assert.equal(nodes.get('#imageHeightInput').value, '1152');
});

test('a removed selection falls back to the available current model', () => {
  const { context, nodes } = loadUi();
  nodes.get('#modelSelect').value = 'Other NovelAI/square';
  nodes.get('#modelSelect').listeners.change();
  nodes.get('#imageWidthInput').value = '1024';
  vm.runInContext(`
    appState.pluginState.available_models = ['Custom name/portrait'];
    renderState();
  `, context);
  assert.equal(nodes.get('#modelSelect').value, 'Custom name/portrait');
  assert.equal(nodes.get('#imageWidthInput').value, '896');
  assert.equal(nodes.get('#imageHeightInput').value, '1152');
});

test('invalid manual pixel sizes stop before a request is submitted', async () => {
  const { context, nodes, calls, messages } = loadUi();
  for (const [width, height] of [
    [63, 1216], [833, 1216], [832.5, 1216], [2112, 832],
    [2048, 2048], ['', 1216], ['Infinity', 1216], ['invalid', 1216],
  ]) {
    nodes.get('#imageWidthInput').value = width;
    nodes.get('#imageHeightInput').value = height;
    await vm.runInContext('submitTask({preventDefault() {}})', context);
    assert.equal(calls.length, 0, `${width}x${height} must not be submitted`);
    assert.equal(messages.at(-1).isError, true);
    assert.equal(nodes.get('#submitBtn').disabled, false);
  }
});

test('other adapters keep the existing ratio and resolution request', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').value = 'NovelAI/chat';
  nodes.get('#modelSelect').listeners.change();
  nodes.get('#imageWidthInput').value = '896';
  nodes.get('#imageHeightInput').value = '1152';
  await vm.runInContext('submitTask({preventDefault() {}})', context);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].payload.aspect_ratio, '3:4');
  assert.equal(calls[0].payload.resolution, '2K');
  assert.equal(Object.hasOwn(calls[0].payload, 'width'), false);
  assert.equal(Object.hasOwn(calls[0].payload, 'height'), false);
});
