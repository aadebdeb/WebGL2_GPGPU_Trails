function createShader(gl, source, type) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) + source);
  }
  return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

function createProgramFromSource(gl, vertexShaderSource, fragmentShaderSource) {
  const program = gl.createProgram();
  gl.attachShader(program, createShader(gl, vertexShaderSource, gl.VERTEX_SHADER));
  gl.attachShader(program, createShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

function createVbo(gl, array, usage) {
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, array, usage !== undefined ? usage : gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return vbo;
}

function createIbo(gl, array) {
  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, array, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  return ibo;
}

function createVao(gl, vboObjs, ibo) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  if (ibo !== undefined) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  }
  vboObjs.forEach((vboObj) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, vboObj.buffer);
    gl.enableVertexAttribArray(vboObj.index);
    gl.vertexAttribPointer(vboObj.index, vboObj.size, gl.FLOAT, false, 0, 0);
  });
  gl.bindVertexArray(null);
  if (ibo !== undefined) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return vao;
}

function createTexture(gl, sizeX, sizeY, internalFormat, format, type) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, sizeX, sizeY, 0, format, type, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

function getUniformLocations(gl, program, keys) {
  const locations = {};
  keys.forEach(key => {
      locations[key] = gl.getUniformLocation(program, key);
  });
  return locations;
}

function setUniformTexture(gl, index, texture, location) {
  gl.activeTexture(gl.TEXTURE0 + index);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(location, index);
}

const FILL_SCREEN_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec2 position;

void main(void) {
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const FILL_SCREEN_VERTICES_POSITION = new Float32Array([
  -1.0, -1.0,
  1.0, -1.0,
  -1.0,  1.0,
  1.0,  1.0
]);

const FILL_SCREEN_VERTICES_INDEX = new Int16Array([
  0, 1, 2,
  3, 2, 1
]);

function createFillScreenVao(gl) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, createIbo(gl, FILL_SCREEN_VERTICES_INDEX));
  gl.bindBuffer(gl.ARRAY_BUFFER, createVbo(gl, FILL_SCREEN_VERTICES_POSITION));
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return vao;
}