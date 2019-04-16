(function() {

  function createTrailFbObj(gl, textureSize) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, textureSize.x, textureSize.y, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, textureSize.x, textureSize.y, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    gl.bindTexture(gl.TEXTURE_2D, velocityTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, velocityTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      positionTexture: positionTexture,
      velocityTexture: velocityTexture,
    };
  }

  const INITIALIZE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

#define PI 3.14159265359

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_velocity;

float random(float x){
  return fract(sin(x * 12.9898) * 43758.5453);
}

vec3 randomInSphere(float v) {
  float z = random(v * 0.42 + 213.23) * 2.0 - 1.0;
  float phi = random(v * 0.19 + 313.98) * PI * 2.0;
  float r = random(v * 0.35 + 192.75);

  float a = pow(r, 1.0 / 3.0);
  float b = sqrt(1.0 - z * z);

  return vec3(a * b * cos(phi), a * b * sin(phi), a * z);
}

void main(void) {
  o_position = 1.0 * randomInSphere(gl_FragCoord.x);
  o_velocity = vec3(0.0);
}
`;

  const UPDATE_FRAGMENT_SHADDER_SOURCE =
`#version 300 es

precision highp float;

layout (location = 0) out vec3 o_position;
layout (location = 1) out vec3 o_velocity;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform float u_time;
uniform float u_deltaTime;
uniform float u_maxSpeed;
uniform float u_maxForce;
uniform float u_boundaryRadius;
uniform float u_noiseScale;

float random(vec4 x){
  return fract(sin(dot(x,vec4(12.9898, 78.233, 39.425, 27.196))) * 43758.5453);
}

float valuenoise(vec4 x) {
  vec4 i = floor(x);
  vec4 f = fract(x);

  vec4 u = f * f * (3.0 - 2.0 * f);

  return mix(
    mix(
      mix(
          mix(random(i + vec4(0.0, 0.0, 0.0, 0.0)), random(i + vec4(1.0, 0.0, 0.0, 0.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 0.0, 0.0)), random(i + vec4(1.0, 1.0, 0.0, 0.0)), u.x),
          u.y
      ),
      mix(
          mix(random(i + vec4(0.0, 0.0, 1.0, 0.0)), random(i + vec4(1.0, 0.0, 1.0, 0.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 1.0, 0.0)), random(i + vec4(1.0, 1.0, 1.0, 0.0)), u.x),
          u.y
      ),
      u.z
    ),
    mix(
      mix(
          mix(random(i + vec4(0.0, 0.0, 0.0, 1.0)), random(i + vec4(1.0, 0.0, 0.0, 1.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 0.0, 1.0)), random(i + vec4(1.0, 1.0, 0.0, 1.0)), u.x),
          u.y
      ),
      mix(
          mix(random(i + vec4(0.0, 0.0, 1.0, 1.0)), random(i + vec4(1.0, 0.0, 1.0, 1.0)), u.x),
          mix(random(i + vec4(0.0, 1.0, 1.0, 1.0)), random(i + vec4(1.0, 1.0, 1.0, 1.0)), u.x),
          u.y
      ),
      u.z
    ),
    u.w
  );
}

float fbm(vec4 x) {
  float sum = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    sum += amp * valuenoise(x);
    amp *= 0.5;
    x *= 2.01;
  }
  return sum * 2.0 - 1.0;
}

float noiseX(vec4 x) {
  return fbm(x * 0.34 + vec4(4324.32, 7553.13, 5417.33, 1484.43));
}

float noiseY(vec4 x) {
  return fbm(x * 0.71 + vec4(1614.43, 8439.32, 4211.93, 8546.29));
}

float noiseZ(vec4 x) {
  return fbm(x * 0.54 + vec4(4342.34, 7569.34, 3812.42, 1589.54));
}

vec3 curlnoise(vec3 x, float time) {
  float e = 0.01;

  vec3 dx = vec3(e, 0.0, 0.0);
  vec3 dy = vec3(0.0, e, 0.0);
  vec3 dz = vec3(0.0, 0.0, e);

  return normalize(vec3(
    noiseZ(vec4(x + dy, time)) - noiseY(vec4(x + dz, time)),
    noiseX(vec4(x + dz, time)) - noiseZ(vec4(x + dx, time)),
    noiseY(vec4(x + dx, time)) - noiseX(vec4(x + dy, time))
  ));
}

vec3 limit(vec3 v, float max) {
  if (length(v) > max) {
    return normalize(v) * max;
  }
  return v;
}

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);

  vec3 nextPosition, nextVelocity;
  if (coord.y == 0) {
    vec3 velocity = texelFetch(u_velocityTexture, coord, 0).xyz;
    vec3 position = texelFetch(u_positionTexture, coord, 0).xyz;

    vec3 acceleration = curlnoise(position * u_noiseScale, u_time * 0.2);

    acceleration = u_maxForce * mix(acceleration, -normalize(position), smoothstep(u_boundaryRadius, u_boundaryRadius * 1.05, length(position)));

    nextVelocity = limit(velocity + u_deltaTime * acceleration, u_maxSpeed);
    nextPosition = position + u_deltaTime * nextVelocity;
  } else {
    nextPosition = texelFetch(u_positionTexture, ivec2(coord.x, coord.y - 1), 0).xyz;
    nextVelocity = texelFetch(u_velocityTexture, ivec2(coord.x, coord.y - 1), 0).xyz;
  }
  o_position = nextPosition;
  o_velocity = nextVelocity;
}

`;

  const RENDER_VERTEX_SHADER_SOURCE =
`#version 300 es

uniform sampler2D u_positionTexture;
uniform mat4 u_vpMatrix;

void main(void) {
  vec3 position = texelFetch(u_positionTexture, ivec2(gl_InstanceID, gl_VertexID), 0).xyz;
  gl_Position = u_vpMatrix * vec4(position, 1.0);
}
`;

  const RENDER_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec4 o_color;

uniform vec3 u_color;
uniform float u_alpha;

void main(void) {
  o_color = vec4(u_color, u_alpha);
}
`;

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const parameters = {
    dynamic: {
      'max speed': 50.0,
      'max force': 30.0,
      'boundary radius': 200.0,
      'noise scale': 0.05,
      'color': [30, 255, 240],
      'alpha': 0.05,
    },
    static: {
      'trail num': 4096,
      'vertex num': 256
    },
    'reset': _ => reset()
  };

  const gui = new dat.GUI();
  const dynamicFolder = gui.addFolder('dynamic parameter');
  dynamicFolder.add(parameters.dynamic, 'max speed', 0.0, 100.0);
  dynamicFolder.add(parameters.dynamic, 'max force', 0.0, 100.0);
  dynamicFolder.add(parameters.dynamic, 'boundary radius', 50.0, 400.0);
  dynamicFolder.add(parameters.dynamic, 'noise scale', 0.0, 0.2).step(0.001);
  dynamicFolder.addColor(parameters.dynamic, 'color');
  dynamicFolder.add(parameters.dynamic, 'alpha', 0.0, 1.0).step(0.001);
  const staticFolder = gui.addFolder('static parameter');
  staticFolder.add(parameters.static, 'trail num', 1, 16384).step(1);
  staticFolder.add(parameters.static, 'vertex num', 1, 1024).step(1);
  gui.add(parameters, 'reset');

  const canvas = document.getElementById('canvas');
  const resizeCanvas = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');

  const initializeProgram = createProgramFromSource(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, INITIALIZE_FRAGMENT_SHADER_SOURCE);
  const updateProgram = createProgramFromSource(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, UPDATE_FRAGMENT_SHADDER_SOURCE);
  const renderProgram = createProgramFromSource(gl, RENDER_VERTEX_SHADER_SOURCE, RENDER_FRAGMENT_SHADER_SOURCE);

  const updateUniforms = getUniformLocations(gl, updateProgram,
    ['u_positionTexture', 'u_velocityTexture', 'u_time', 'u_deltaTime', 'u_maxSpeed', 'u_maxForce', 'u_boundaryRadius', 'u_noiseScale']);
  const renderUniforms = getUniformLocations(gl, renderProgram, ['u_positionTexture', 'u_vpMatrix', 'u_color', 'u_alpha']);

  const fillScreenVao = createFillScreenVao(gl);
  const renderToFillScreen = () => {
    gl.bindVertexArray(fillScreenVao);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
  };

  let requestId = null;
  const reset = () => {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const trailNum = parameters.static['trail num'];
    const vertexNum = parameters.static['vertex num'];

    const trailTextureSize = new Vector2(trailNum, vertexNum);
    let trailFbObjR = createTrailFbObj(gl, trailTextureSize);
    let trailFbObjW = createTrailFbObj(gl, trailTextureSize);
    const swapTrailFbObj = () => {
      const tmp = trailFbObjR;
      trailFbObjR = trailFbObjW;
      trailFbObjW = tmp;
    };

    const initializeTrails = () => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, trailFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, trailTextureSize.x, trailTextureSize.y);
      gl.useProgram(initializeProgram);
      renderToFillScreen();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapTrailFbObj();
    };

    const updateTrails = (elapsedTime, deltaTime) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, trailFbObjW.framebuffer);
      gl.viewport(0.0, 0.0, trailTextureSize.x, trailTextureSize.y);
      gl.useProgram(updateProgram);
      setUniformTexture(gl, 0, trailFbObjR.positionTexture, updateUniforms['u_positionTexture']);
      setUniformTexture(gl, 1, trailFbObjR.velocityTexture, updateUniforms['u_velocityTexture']);
      gl.uniform1f(updateUniforms['u_time'], elapsedTime);
      gl.uniform1f(updateUniforms['u_deltaTime'], deltaTime);
      gl.uniform1f(updateUniforms['u_maxSpeed'], parameters.dynamic['max speed']);
      gl.uniform1f(updateUniforms['u_maxForce'], parameters.dynamic['max force']);
      gl.uniform1f(updateUniforms['u_boundaryRadius'], parameters.dynamic['boundary radius']);
      gl.uniform1f(updateUniforms['u_noiseScale'], parameters.dynamic['noise scale']);
      renderToFillScreen();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      swapTrailFbObj();
    };

    const renderTrails = () => {
      const viewMatrix = Matrix4.inverse(Matrix4.lookAt(
        new Vector3(0.0, 0.0, 500.0),
        Vector3.zero,
        new Vector3(0.0, 1.0, 0.0)
      ));
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60, 0.01, 1000.0);
      const vpMatrix = Matrix4.mul(viewMatrix, projectionMatrix);

      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.useProgram(renderProgram);
      setUniformTexture(gl, 0, trailFbObjR.positionTexture, renderUniforms['u_positionTexture']);
      gl.uniformMatrix4fv(renderUniforms['u_vpMatrix'], false, vpMatrix.elements);
      gl.uniform3fv(renderUniforms['u_color'], parameters.dynamic['color'].map(v => v / 255.0));
      gl.uniform1f(renderUniforms['u_alpha'], parameters.dynamic['alpha']);
      gl.drawArraysInstanced(gl.LINE_STRIP, 0, vertexNum, trailNum);
      gl.disable(gl.BLEND);
    };

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    let elapsedTime = 0.0;
    let previousTime = performance.now();
    const render = () => {
      stats.update();

      const currentTime = performance.now();
      const deltaTime = Math.min(0.05, (currentTime - previousTime) * 0.001);
      elapsedTime += deltaTime;
      previousTime = currentTime;

      updateTrails(elapsedTime, deltaTime);
      renderTrails();

      requestId = requestAnimationFrame(render);
    }
    initializeTrails();
    render();
  };
  reset();
}());