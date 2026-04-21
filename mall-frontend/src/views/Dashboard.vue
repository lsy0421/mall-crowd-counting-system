<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'

/** 实时画面人数（来自 /api/stats/current 的 person_count） */
const personCount = ref(0)

/** 当前系统帧率（来自 /api/stats/current 的 fps） */
const systemFps = ref(0)

/** 图表挂载容器（优先用 ref 获取 DOM） */
const chartContainerRef = ref(null)

let chartInstance = null
/** 实时统计轮询：2s */
let statsPollTimer = null
/** 历史曲线轮询：60s */
let historyPollTimer = null

/**
 * 将 ISO 时间戳格式化为横轴可读文本（月-日 时:分）
 */
function formatAxisLabel(isoString) {
  const d = new Date(isoString)
  if (Number.isNaN(d.getTime())) return isoString
  const M = String(d.getMonth() + 1).padStart(2, '0')
  const D = String(d.getDate()).padStart(2, '0')
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  return `${M}-${D} ${h}:${m}`
}

/**
 * 拉取当前统计并更新界面上的 ref
 */
async function fetchCurrentStats() {
  try {
    const { data } = await axios.get('/api/stats/current')
    if (data && typeof data === 'object' && 'error' in data && data.error) {
      return
    }
    const pc = data?.person_count
    const fp = data?.fps
    personCount.value = pc != null && pc !== '' ? Number(pc) : 0
    systemFps.value = fp != null && fp !== '' ? Number(fp) : 0
  } catch (err) {
    console.warn('[Dashboard] fetchCurrentStats 失败:', err)
  }
}

/**
 * 拉取 24h 历史数据并更新折线图（需已 initChart）
 */
async function fetchHistoryStats() {
  try {
    const { data } = await axios.get('/api/stats/history')
    if (!Array.isArray(data)) return

    const xData = data.map((row) => formatAxisLabel(row.timestamp))
    const yData = data.map((row) => Number(row.total_count) || 0)

    if (!chartInstance) return

    chartInstance.setOption(
      {
        grid: {
          left: 52,
          right: 20,
          top: 36,
          bottom: 28,
          containLabel: true,
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross', label: { backgroundColor: '#64748b' } },
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: xData,
          axisLine: { lineStyle: { color: '#cbd5e1' } },
          axisLabel: { color: '#64748b', fontSize: 11, rotate: xData.length > 12 ? 35 : 0 },
        },
        yAxis: {
          type: 'value',
          name: '人数',
          minInterval: 1,
          splitLine: { lineStyle: { type: 'dashed', color: '#e2e8f0' } },
          axisLabel: { color: '#64748b' },
          nameTextStyle: { color: '#64748b' },
        },
        series: [
          {
            name: '人流量',
            type: 'line',
            data: yData,
            smooth: true,
            showSymbol: yData.length <= 48,
            symbol: 'circle',
            symbolSize: 7,
            lineStyle: {
              width: 2.5,
              color: '#3b82f6',
              shadowColor: 'rgba(59, 130, 246, 0.35)',
              shadowBlur: 8,
            },
            itemStyle: {
              color: '#2563eb',
              borderColor: '#fff',
              borderWidth: 1,
            },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: 'rgba(59, 130, 246, 0.42)' },
                { offset: 0.5, color: 'rgba(59, 130, 246, 0.12)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.02)' },
              ]),
            },
          },
        ],
      },
      { notMerge: true },
    )
  } catch (err) {
    console.warn('[Dashboard] fetchHistoryStats 失败:', err)
  }
}

/**
 * 在 chart-container 上初始化 ECharts 实例（仅创建实例，数据由 fetchHistoryStats 填充）
 */
function initChart() {
  nextTick(() => {
    const el = chartContainerRef.value
    if (!el) {
      console.warn('[Dashboard] chart-container 未就绪')
      return
    }
    if (chartInstance) {
      chartInstance.dispose()
      chartInstance = null
    }
    chartInstance = echarts.init(el, null, { renderer: 'canvas' })
    fetchHistoryStats()
  })
}

function onWindowResize() {
  chartInstance?.resize()
}

onMounted(() => {
  fetchCurrentStats()
  statsPollTimer = window.setInterval(fetchCurrentStats, 2000)

  initChart()
  historyPollTimer = window.setInterval(fetchHistoryStats, 60_000)

  window.addEventListener('resize', onWindowResize)
})

onUnmounted(() => {
  if (statsPollTimer != null) {
    clearInterval(statsPollTimer)
    statsPollTimer = null
  }
  if (historyPollTimer != null) {
    clearInterval(historyPollTimer)
    historyPollTimer = null
  }
  window.removeEventListener('resize', onWindowResize)
  chartInstance?.dispose()
  chartInstance = null
})
</script>

<template>
  <el-container class="dashboard" direction="vertical">
    <el-header class="dashboard-header" height="56px">
      <h1 class="dashboard-title">商场智能人流与热点分析系统</h1>
    </el-header>

    <el-main class="dashboard-main">
      <el-row :gutter="20" class="dashboard-row">
        <!-- 左：视频监控（约 14/24） -->
        <el-col :xs="24" :lg="14">
          <el-card class="panel video-panel" shadow="hover">
            <template #header>
              <span class="panel-title">实时视频监控</span>
            </template>
            <div class="video-wrap">
              <img
                class="video-stream"
                :src="'/api/video_feed'"
                alt="实时视频流"
              />
            </div>
          </el-card>
        </el-col>

        <!-- 右：数据看板（约 10/24） -->
        <el-col :xs="24" :lg="10">
          <div class="right-stack">
            <el-row :gutter="16" class="stats-row">
              <el-col :span="12">
                <el-card class="stat-card" shadow="hover">
                  <el-statistic
                    title="实时画面人数"
                    :value="personCount"
                    suffix="人"
                  />
                </el-card>
              </el-col>
              <el-col :span="12">
                <el-card class="stat-card" shadow="hover">
                  <el-statistic
                    title="当前系统帧率 (FPS)"
                    :value="systemFps"
                    :precision="1"
                  />
                </el-card>
              </el-col>
            </el-row>

            <el-card class="panel chart-panel" shadow="hover">
              <template #header>
                <span class="panel-title">近 24 小时人流趋势</span>
              </template>
              <div
                id="chart-container"
                ref="chartContainerRef"
                class="chart-container"
              />
            </el-card>
          </div>
        </el-col>
      </el-row>
    </el-main>
  </el-container>
</template>

<style scoped>
.dashboard {
  min-height: 100vh;
  background: linear-gradient(165deg, #f0f4f8 0%, #e8eef5 45%, #f5f7fa 100%);
  color: var(--el-text-color-primary);
}

.dashboard :deep(.el-main) {
  overflow: visible;
}

.dashboard-header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 24px;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--el-border-color-lighter);
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
}

.dashboard-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: #1e293b;
}

.dashboard-main {
  flex: 1;
  padding: 20px 24px 28px;
  max-width: 1920px;
  margin: 0 auto;
  width: 100%;
  box-sizing: border-box;
}

.dashboard-row {
  align-items: stretch;
}

.panel {
  border-radius: 12px;
  border: 1px solid var(--el-border-color-lighter);
  background: #fff;
}

.panel :deep(.el-card__header) {
  padding: 14px 18px;
  border-bottom: 1px solid var(--el-border-color-extra-light);
}

.panel-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: #334155;
}

.video-panel {
  height: 100%;
  min-height: 420px;
}

.video-panel :deep(.el-card__body) {
  padding: 12px;
  height: calc(100% - 54px);
  min-height: 360px;
}

.video-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 320px;
  border-radius: 8px;
  overflow: hidden;
  background: #0f172a;
}

.video-stream {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.right-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  min-height: 420px;
}

.stats-row {
  flex-shrink: 0;
}

.stat-card {
  border-radius: 12px;
  border: 1px solid var(--el-border-color-lighter);
  background: #fff;
}

.stat-card :deep(.el-card__body) {
  padding: 18px 16px;
}

.stat-card :deep(.el-statistic__head) {
  font-size: 0.85rem;
  color: #64748b;
}

.stat-card :deep(.el-statistic__number) {
  font-weight: 700;
  color: #0f172a;
}

.chart-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.chart-panel :deep(.el-card__body) {
  flex: 1;
  padding: 12px 16px 16px;
  display: flex;
  flex-direction: column;
}

.chart-container {
  flex: 1;
  width: 100%;
  min-height: 400px;
  border-radius: 8px;
  background: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%);
  border: 1px dashed var(--el-border-color);
}

@media (max-width: 1199px) {
  .dashboard-row .el-col {
    margin-bottom: 16px;
  }

  .right-stack {
    min-height: auto;
  }

  .video-panel :deep(.el-card__body) {
    min-height: 280px;
  }

  .video-wrap {
    min-height: 240px;
  }
}
</style>
