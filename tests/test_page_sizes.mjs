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
    'negativePromptField', 'useDefaultNegativePrompt', 'negativePromptInput', 'negativePromptHint',
    'novelaiOptionsPanel', 'novelaiModeSelect', 'novelaiParametersInput',
    'novelaiMaskField', 'novelaiMaskInput', 'novelaiMaskList', 'novelaiCharacterInput',
    'novelaiCharacterList', 'novelaiVibeInput', 'novelaiVibeFileInput',
    'novelaiVibeVariantField', 'novelaiVibeVariantSelect',
    'referenceInput', 'referenceImageLabel', 'uploadList',
  ]) {
    let value = '';
    const node = {
      hidden: false,
      disabled: false,
      required: false,
      checked: id === 'useDefaultNegativePrompt',
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
    JSON,
  });
  vm.runInContext(source, context);
  vm.runInContext(`
    appState.pluginState = {
      current_model: 'Custom name/portrait',
      available_models: ['Custom name/portrait', 'NovelAI/chat', 'Other NovelAI/square', 'OpenAI/gpt-image-2', 'Gemini/plain'],
      model_settings: {
        'Custom name/portrait': {adapter_type: 'novelai', supports_negative_prompt: true, width: 896, height: 1152},
        'NovelAI/chat': {adapter_type: 'openai_chat', supports_negative_prompt: true, negative_prompt_mode: 'parameter'},
        'Other NovelAI/square': {adapter_type: 'novelai', supports_negative_prompt: true, width: 512, height: 512},
        'OpenAI/gpt-image-2': {adapter_type: 'openai', supports_negative_prompt: true, negative_prompt_mode: 'prompt'},
        'Gemini/plain': {adapter_type: 'gemini'},
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

test('negative prompt defaults are inherited without adding a request override', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#negativePromptField').hidden, false);
  assert.equal(nodes.get('#useDefaultNegativePrompt').checked, true);
  assert.equal(nodes.get('#negativePromptInput').disabled, true);
  nodes.get('#negativePromptInput').value = 'an earlier override';
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls.length, 1);
  assert.equal(Object.hasOwn(calls[0].payload, 'negative_prompt'), false);
});

test('negative prompt overrides and an explicit empty value are submitted per task', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  nodes.get('#useDefaultNegativePrompt').checked = false;
  nodes.get('#useDefaultNegativePrompt').listeners.change();
  assert.equal(nodes.get('#negativePromptInput').disabled, false);
  nodes.get('#negativePromptInput').value = ' blur, watermark ';
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls[0].payload.negative_prompt, 'blur, watermark');

  nodes.get('#negativePromptInput').value = '   ';
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls[1].payload.negative_prompt, '');

  nodes.get('#useDefaultNegativePrompt').checked = true;
  nodes.get('#useDefaultNegativePrompt').listeners.change();
  assert.equal(nodes.get('#negativePromptInput').disabled, true);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(Object.hasOwn(calls[2].payload, 'negative_prompt'), false);
});

test('model switches hide unsupported negative prompts and preserve an unsubmitted override', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  nodes.get('#useDefaultNegativePrompt').checked = false;
  nodes.get('#useDefaultNegativePrompt').listeners.change();
  nodes.get('#negativePromptInput').value = 'watermark';

  nodes.get('#modelSelect').value = 'Gemini/plain';
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#negativePromptField').hidden, true);
  assert.equal(nodes.get('#useDefaultNegativePrompt').disabled, true);
  assert.equal(nodes.get('#negativePromptInput').disabled, true);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(Object.hasOwn(calls[0].payload, 'negative_prompt'), false);

  nodes.get('#modelSelect').value = 'Custom name/portrait';
  nodes.get('#modelSelect').listeners.change();
  vm.runInContext('renderState()', context);
  await vm.runInContext('refreshLocalizedUi()', context);
  assert.equal(nodes.get('#negativePromptField').hidden, false);
  assert.equal(nodes.get('#useDefaultNegativePrompt').disabled, false);
  assert.equal(nodes.get('#useDefaultNegativePrompt').checked, false);
  assert.equal(nodes.get('#negativePromptInput').disabled, false);
  assert.equal(nodes.get('#negativePromptInput').value, 'watermark');
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls[1].payload.negative_prompt, 'watermark');
});

test('NovelAI controls follow the selected model and preserve empty default overrides', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#novelaiOptionsPanel').hidden, false);
  assert.equal(nodes.get('#referenceInput').multiple, false);
  assert.equal(nodes.get('#referenceInput').disabled, false);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(Object.hasOwn(calls[0].payload, 'novelai'), false);

  nodes.get('#novelaiModeSelect').value = 'generate';
  nodes.get('#novelaiModeSelect').listeners.change();
  assert.equal(nodes.get('#referenceInput').disabled, true);
  assert.equal(nodes.get('#novelaiMaskField').hidden, true);
  nodes.get('#modelSelect').value = 'NovelAI/chat';
  nodes.get('#modelSelect').listeners.change();
  assert.equal(nodes.get('#novelaiOptionsPanel').hidden, true);
  assert.equal(nodes.get('#referenceInput').multiple, true);
});

test('NovelAI sampling, characters, negative override and source tokens are submitted together', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  nodes.get('#novelaiParametersInput').value = JSON.stringify({
    steps: 32, sampler: 'k_euler', seed: 123, scale: 4.5,
    characters: [{prompt: 'red hair', negative_prompt: 'hat', position: 'B3'}],
  });
  nodes.get('#novelaiModeSelect').value = 'img2img';
  nodes.get('#useDefaultNegativePrompt').checked = false;
  nodes.get('#negativePromptInput').value = 'watermark';
  vm.runInContext("appState.uploads = [{token:'source'}]", context);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].payload.reference_tokens, ['source']);
  assert.deepEqual(calls[0].payload.novelai.i2i, {});
  assert.equal(calls[0].payload.novelai.action, 'img2img');
  assert.equal(calls[0].payload.novelai.steps, 32);
  assert.equal(calls[0].payload.novelai.characters[0].position, 'B3');
  assert.equal(calls[0].payload.negative_prompt, 'watermark');
});

test('inpainting masks and precise references stay separate from image-to-image sources', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  vm.runInContext(`
    appState.uploads = [{token:'source'}];
    appState.novelaiMask = {token:'mask'};
    appState.novelaiReferences = [{token:'character',type:'character&style',strength:0.7,fidelity:0.9}];
  `, context);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].payload.reference_tokens, ['source']);
  assert.equal(calls[0].payload.novelai_mask_token, 'mask');
  assert.deepEqual(calls[0].payload.novelai.inpaint, {});
  assert.deepEqual(calls[0].payload.novelai_character_references, [
    {token:'character',type:'character&style',strength:0.7,fidelity:0.9},
  ]);

  nodes.get('#novelaiModeSelect').value = 'generate';
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.deepEqual(calls[1].payload.reference_tokens, []);
  assert.equal(Object.hasOwn(calls[1].payload, 'novelai_mask_token'), false);
  assert.equal(calls[1].payload.novelai.action, 'generate');
});

test('encoded Vibe accepts Base64 and JSON text without calling encoding tools', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  nodes.get('#novelaiVibeInput').value = 'ZW5jb2RlZC12aWJl';
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.deepEqual(calls[0].payload.novelai.vibe_references, [
    {encoding:'ZW5jb2RlZC12aWJl',information_extracted:1,strength:0.6},
  ]);
  nodes.get('#novelaiParametersInput').value = JSON.stringify({vibe_references:[{encoding:'first',strength:0.2}]});
  nodes.get('#novelaiVibeInput').value = JSON.stringify([{encoding:'second',strength:0.8,information_extracted:0.9}]);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls[1].payload.novelai.vibe_references.length, 2);
  assert.equal(calls[1].payload.novelai.vibe_references[1].strength, 0.8);
  assert.ok(calls.every((call) => call.endpoint === 'page/generate'));

  const input = nodes.get('#novelaiVibeFileInput');
  await input.listeners.change({target:{files:[{size:12,text:async () => 'imported-vibe'}],value:'selected'}});
  assert.equal(nodes.get('#novelaiVibeInput').value, 'imported-vibe');
});

test('invalid advanced JSON, incompatible modes and excessive references stop before submission', async () => {
  const { context, nodes, calls, messages } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  for (const value of ['{', '[]', 'null', '{"i2i":true}', '{"characters":[null]}', '{"characters":[' + Array(7).fill('{}').join(',') + ']}', '{"inpaint":{}}', '{"i2i":{},"inpaint":{"mask":"file"}}', '{"action":"generate","i2i":{}}']) {
    nodes.get('#novelaiParametersInput').value = value;
    await vm.runInContext('submitTask({preventDefault() {}})', context);
    assert.equal(calls.length, 0, value);
    assert.equal(messages.at(-1).isError, true);
  }
  nodes.get('#novelaiParametersInput').value = '{}';
  nodes.get('#novelaiVibeInput').value = '[{"encoding":"vibe"}]';
  vm.runInContext("appState.novelaiReferences = [{token:'character'}]", context);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls.length, 0);
});

test('role-based uploads retain duplicate precise references, edits and deletion', async () => {
  const { context, nodes } = loadUi();
  nodes.get('#modelSelect').listeners.change();
  vm.runInContext(`appState.bridge = {upload:async (_endpoint,file) => ({token:file.name,filename:file.name,size:4})}`, context);
  await vm.runInContext("uploadReferences([{name:'source'}])", context);
  await vm.runInContext("uploadReferences([{name:'mask'}], 'mask')", context);
  await vm.runInContext("uploadReferences([{name:'same'},{name:'same'}], 'character')", context);
  assert.equal(vm.runInContext('appState.uploads.length', context), 1);
  assert.equal(vm.runInContext('appState.novelaiMask.token', context), 'mask');
  assert.equal(vm.runInContext('appState.novelaiReferences.length', context), 2);
  nodes.get('#novelaiCharacterList').listeners.change({target:{dataset:{novelaiReferenceIndex:'1',novelaiReferenceKey:'strength'},value:'0.25'}});
  assert.equal(vm.runInContext('appState.novelaiReferences[1].strength', context), 0.25);
  nodes.get('#novelaiCharacterList').listeners.click({target:{dataset:{novelaiReferenceRemove:'0'}}});
  assert.equal(vm.runInContext('appState.novelaiReferences.length', context), 1);
  assert.equal(vm.runInContext('appState.novelaiReferences[0].strength', context), 0.25);
  nodes.get('#novelaiMaskList').listeners.click({target:{dataset:{novelaiMaskRemove:''}}});
  assert.equal(vm.runInContext('appState.novelaiMask', context), null);
});

test('other adapters never receive retained NovelAI advanced state', async () => {
  const { context, nodes, calls } = loadUi();
  nodes.get('#novelaiParametersInput').value = '{invalid retained json';
  nodes.get('#novelaiVibeInput').value = 'vibe';
  vm.runInContext(`
    appState.uploads = [{token:'ordinary'}];
    appState.novelaiMask = {token:'mask'};
    appState.novelaiReferences = [{token:'character'}];
  `, context);
  nodes.get('#modelSelect').value = 'NovelAI/chat';
  nodes.get('#modelSelect').listeners.change();
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].payload.reference_tokens, ['ordinary']);
  for (const key of ['novelai','novelai_mask_token','novelai_character_references']) {
    assert.equal(Object.hasOwn(calls[0].payload, key), false);
  }
});

test('naiv4vibe import uses model-specific encoded features and ignores source images', async () => {
  const { context, nodes, calls } = loadUi();
  vm.runInContext("appState.pluginState.model_settings['Custom name/portrait'].model = 'nai-diffusion-4-5-full'", context);
  nodes.get('#modelSelect').listeners.change();
  const document = {
    identifier:'novelai-vibe-transfer',version:1,type:'image',
    image:'RAW_SOURCE_IMAGE_MUST_NOT_BE_USED',thumbnail:'RAW_THUMBNAIL_MUST_NOT_BE_USED',
    importInfo:{model:'nai-diffusion-4-5-full',information_extracted:0.99,strength:0.7},
    encodings:{
      'v4-5full':{['a'.repeat(64)]:{encoding:'ZW5jb2RlZC1mdWxs',params:{information_extracted:0.5}}},
      'v4-5curated':{other:{encoding:'ZW5jb2RlZC1jdXJhdGVk',params:{information_extracted:1}}},
    },
  };
  const input = nodes.get('#novelaiVibeFileInput');
  await input.listeners.change({target:{files:[{name:'reference.naiv4vibe',size:1000,text:async () => JSON.stringify(document)}],value:'selected'}});
  assert.deepEqual(JSON.parse(nodes.get('#novelaiVibeInput').value), [
    {encoding:'ZW5jb2RlZC1mdWxs',information_extracted:0.5,strength:0.7},
  ]);
  assert.equal(nodes.get('#novelaiVibeVariantField').hidden, true);
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls[0].payload.novelai.vibe_references[0].encoding, 'ZW5jb2RlZC1mdWxs');
  assert.equal(JSON.stringify(calls[0].payload).includes('RAW_SOURCE'), false);
});

test('naiv4vibe import requires an explicit variant and blocks mismatched models', async () => {
  const { context, nodes, calls, messages } = loadUi();
  vm.runInContext(`
    appState.pluginState.model_settings['Custom name/portrait'].model = 'nai-diffusion-4-5-full';
    appState.pluginState.model_settings['Other NovelAI/square'].model = 'nai-diffusion-4-5-curated';
  `, context);
  nodes.get('#modelSelect').listeners.change();
  const document = {
    identifier:'novelai-vibe-transfer',version:1,type:'image',
    encodings:{'v4-5full':{
      first:{encoding:'Zmlyc3QtZW5jb2Rpbmc=',params:{information_extracted:1}},
      second:{encoding:'c2Vjb25kLWVuY29kaW5n',params:{information_extracted:1}},
    }},
  };
  await nodes.get('#novelaiVibeFileInput').listeners.change({target:{files:[{name:'variants.naiv4vibe',size:1000,text:async () => JSON.stringify(document)}],value:'selected'}});
  assert.equal(nodes.get('#novelaiVibeVariantField').hidden, false);
  assert.equal(nodes.get('#novelaiVibeInput').value, '');
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls.length, 0);
  assert.equal(messages.at(-1).isError, true);

  const select = nodes.get('#novelaiVibeVariantSelect');
  select.value = '1';
  select.listeners.change({target:select});
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls[0].payload.novelai.vibe_references[0].encoding, 'c2Vjb25kLWVuY29kaW5n');
  nodes.get('#modelSelect').value = 'Other NovelAI/square';
  nodes.get('#modelSelect').listeners.change();
  await vm.runInContext('submitTask({preventDefault() {}})', context);
  assert.equal(calls.length, 1);
  assert.equal(messages.at(-1).isError, true);
});

test('OpenAI Images and Chat submit positive and negative prompts in the same task', async () => {
  for (const model of ['OpenAI/gpt-image-2', 'NovelAI/chat']) {
    const { context, nodes, calls } = loadUi();
    nodes.get('#modelSelect').value = model;
    nodes.get('#modelSelect').listeners.change();
    assert.equal(nodes.get('#negativePromptField').hidden, false);
    assert.equal(nodes.get('#novelaiOptionsPanel').hidden, true);
    nodes.get('#useDefaultNegativePrompt').checked = false;
    nodes.get('#useDefaultNegativePrompt').listeners.change();
    assert.equal(nodes.get('#negativePromptInput').disabled, false);
    nodes.get('#promptInput').value = 'white cat in sunlight';
    nodes.get('#negativePromptInput').value = 'watermark, text';
    nodes.get('#novelaiParametersInput').value = '{retained invalid NAI JSON';
    await vm.runInContext('submitTask({preventDefault() {}})', context);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].endpoint, 'page/generate');
    assert.equal(calls[0].payload.prompt, 'white cat in sunlight');
    assert.equal(calls[0].payload.negative_prompt, 'watermark, text');
    assert.equal(calls[0].payload.model, model);
    assert.equal(Object.hasOwn(calls[0].payload, 'novelai'), false);
  }
});

test('OpenAI-compatible tasks preserve inherited and explicitly cleared negative prompts', async () => {
  for (const model of ['OpenAI/gpt-image-2', 'NovelAI/chat']) {
    const { context, nodes, calls } = loadUi();
    nodes.get('#modelSelect').value = model;
    nodes.get('#modelSelect').listeners.change();
    await vm.runInContext('submitTask({preventDefault() {}})', context);
    assert.equal(Object.hasOwn(calls[0].payload, 'negative_prompt'), false);
    nodes.get('#useDefaultNegativePrompt').checked = false;
    nodes.get('#negativePromptInput').value = '';
    nodes.get('#useDefaultNegativePrompt').listeners.change();
    await vm.runInContext('submitTask({preventDefault() {}})', context);
    assert.equal(calls[1].payload.negative_prompt, '');
    assert.equal(calls[1].payload.prompt, 'cat ears');
  }
});

test('negative prompt hints distinguish semantic constraints from extension passthrough', () => {
  const { nodes } = loadUi();
  nodes.get('#modelSelect').value = 'OpenAI/gpt-image-2';
  nodes.get('#modelSelect').listeners.change();
  assert.match(nodes.get('#negativePromptHint').textContent, /语义约束/);
  assert.match(nodes.get('#negativePromptHint').textContent, /不使用独立负向通道/);
  nodes.get('#modelSelect').value = 'NovelAI/chat';
  nodes.get('#modelSelect').listeners.change();
  assert.match(nodes.get('#negativePromptHint').textContent, /独立 negative_prompt 字段/);
  assert.match(nodes.get('#negativePromptHint').textContent, /明确支持该扩展/);
  nodes.get('#modelSelect').value = 'Custom name/portrait';
  nodes.get('#modelSelect').listeners.change();
  assert.match(nodes.get('#negativePromptHint').textContent, /同一次生图/);
});
