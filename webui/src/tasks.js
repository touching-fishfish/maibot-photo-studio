(() => {
  "use strict";
  const U = window.PhotoUI;
  const { $, element, button } = U;
  let filters = { status: "", origin: "", offset: 0, limit: 12 };
  let galleryFilters = {};
  let items = [];
  let total = 0;
  let task = null;
  let generation = 0;
  let detailGeneration = 0;
  let loading = false;
  let pollingTimer = null;
  let pollFailures = 0;
  let disposed = false;

  function taskId(value) {
    const id = value.task_id || value.id;
    if (typeof id !== "string" || !id) throw new Error("任务缺少有效任务号。");
    return id;
  }
  function taskLabel(value) { return U.operations[value.operation] || U.operations[value.kind] || "其它处理任务"; }
  function originLabel(value) { return U.origins[value.origin] || "其它来源（只读）"; }
  function canMutate(value, operation) {
    return value.origin === "webui" && (operation === "cancel" ? value.can_cancel === true : value.can_retry === true);
  }
  function updatePagination() {
    $("previous-task-page").disabled = loading || U.busy || filters.offset === 0;
    $("next-task-page").disabled = loading || U.busy || filters.offset + filters.limit >= total;
    $("refresh-tasks").disabled = loading || U.busy;
  }
  function updateDetailControls() {
    U.unavailable($("task-cancel"), !task || !canMutate(task, "cancel"));
    U.unavailable($("task-retry"), !task || !canMutate(task, "retry"));
    U.unavailable($("task-view-asset"), !task || !task.asset_id);
    $("task-close").disabled = U.busy;
    $("task-refresh-detail").disabled = loading || U.busy;
  }
  function renderTasks() {
    const list = $("task-list");
    list.replaceChildren();
    $("task-empty").hidden = items.length !== 0;
    $("task-table-wrap").hidden = items.length === 0;
    for (const entry of items) {
      const row = element("tr");
      const identity = element("td");
      identity.append(element("p", { class: "mono" }, taskId(entry)), element("p", { class: "help" }, originLabel(entry)));
      const operation = element("td", {}, taskLabel(entry));
      const state = element("td");
      state.append(element("span", { class: "badge" }, U.taskStatuses[entry.status] || "未知状态"), element("p", { class: "help" }, entry.paid_request_started ? "模型请求已开始，可能已计费" : "尚未标记模型请求开始"));
      if (entry.error) state.append(element("p", { class: "task-error" }, String(entry.error)));
      const assetCell = element("td");
      if (entry.asset_id) assetCell.append(element("p", { class: "mono" }, `素材：${entry.asset_id}`));
      if (entry.asset_status) assetCell.append(element("p", { class: "help" }, `素材状态：${U.assetStatuses[entry.asset_status] || "未知状态"}`));
      if (entry.batch_id) assetCell.append(element("p", { class: "mono" }, `批次：${entry.batch_id}`));
      if (!entry.asset_id && !entry.batch_id) assetCell.append(element("span", { class: "help" }, "暂未关联素材"));
      const created = element("td", {}, U.formatTime(entry.created_at));
      const actions = element("td");
      const toolbar = element("div", { class: "toolbar" });
      toolbar.append(button("详情", () => openTask(taskId(entry)), { disabled: U.busy }));
      if (entry.origin !== "webui") toolbar.append(element("span", { class: "help" }, "只读"));
      else {
        const cancel = button("取消", () => mutateTask("cancel", entry), { "data-mutate": "", class: "danger" });
        const retry = button("重试", () => mutateTask("retry", entry), { "data-mutate": "" });
        U.unavailable(cancel, !canMutate(entry, "cancel"));
        U.unavailable(retry, !canMutate(entry, "retry"));
        toolbar.append(cancel, retry);
      }
      actions.append(toolbar);
      row.append(identity, operation, state, assetCell, created, actions);
      list.append(row);
    }
    $("task-page-label").textContent = total ? `第 ${Math.floor(filters.offset / filters.limit) + 1} 页 · 共 ${total} 条任务` : "共 0 条任务";
    updatePagination();
  }
  function metadata(label, value) {
    $("task-metadata").append(element("dt", {}, label), element("dd", {}, value));
  }
  function renderDetail() {
    if (!task) return;
    $("task-metadata").replaceChildren();
    metadata("任务号", taskId(task));
    metadata("处理方式", taskLabel(task));
    metadata("来源", originLabel(task));
    metadata("任务状态", U.taskStatuses[task.status] || "未知状态");
    metadata("模型请求", task.paid_request_started ? "已经开始，可能已产生费用" : "尚未标记开始；不代表重试免费");
    metadata("素材号", task.asset_id || "暂无");
    metadata("素材状态", task.asset_status ? U.assetStatuses[task.asset_status] || "未知状态" : "暂无");
    metadata("批次号", task.batch_id || "不是批量任务");
    metadata("创建时间", U.formatTime(task.created_at));
    if (task.updated_at) metadata("更新时间", U.formatTime(task.updated_at));
    $("task-detail-error").hidden = !task.error;
    $("task-detail-error").textContent = task.error ? String(task.error) : "";
    $("task-readonly").hidden = task.origin === "webui";
    $("task-paid-note").hidden = !task.paid_request_started;
    updateDetailControls();
  }
  async function readDetail(id) {
    const current = ++detailGeneration;
    const result = await U.query("tasks", "detail", { task_id: id });
    if (current !== detailGeneration || disposed) return null;
    if (!U.plainObject(result.task) || taskId(result.task) !== id) throw new Error("任务详情响应与查询不一致。");
    task = result.task;
    renderDetail();
    return task;
  }
  async function openTask(id) {
    if (U.busy) return;
    try { if (await readDetail(id)) $("task-dialog").showModal(); }
    catch (error) { U.notify(U.message(error), true); }
  }
  function closeTask() {
    if (U.busy) return;
    detailGeneration += 1;
    task = null;
    $("task-dialog").close();
  }
  function schedulePoll() {
    if (pollingTimer !== null) clearTimeout(pollingTimer);
    pollingTimer = null;
    if (disposed || !$("task-poll").checked) { $("poll-status").textContent = "自动刷新已关闭。"; return; }
    if (document.visibilityState === "hidden") { $("poll-status").textContent = "页面已隐藏，轮询暂停。"; return; }
    const interval = Math.min(60000, 5000 * (2 ** pollFailures));
    $("poll-status").textContent = pollFailures ? `查询失败后退避，约 ${interval / 1000} 秒后再查。` : "页面可见时每 5 秒只读刷新。";
    pollingTimer = setTimeout(() => {
      pollingTimer = null;
      if (disposed || document.visibilityState === "hidden" || !$("task-poll").checked) return;
      if (U.busy || loading) { schedulePoll(); return; }
      void refreshTasks(true);
    }, interval);
  }
  async function refreshTasks(poll = false) {
    if (disposed || document.visibilityState === "hidden") { schedulePoll(); return; }
    if (loading) return;
    const current = ++generation;
    loading = true;
    $("task-list").setAttribute("aria-busy", "true");
    updatePagination();
    try {
      // 失败的查询可指数退避；轮询不重试 action，也不创建新任务。
      const result = await U.query("tasks", "list", U.clone(filters), { retries: poll ? 0 : 2 });
      if (current !== generation || disposed) return;
      if (!Array.isArray(result.items) || !Number.isInteger(result.total) || result.total < 0) throw new Error("任务列表结构不符合约定。");
      result.items.forEach(taskId);
      items = result.items;
      total = result.total;
      if (!items.length && filters.offset >= total && filters.offset > 0) {
        filters.offset = Math.max(0, Math.floor(Math.max(0, total - 1) / filters.limit) * filters.limit);
        loading = false;
        return refreshTasks(poll);
      }
      renderTasks();
      if (task && $("task-dialog").open) await readDetail(taskId(task));
      pollFailures = 0;
    } catch (error) {
      if (current === generation) {
        pollFailures = Math.min(4, pollFailures + 1);
        $("task-page-label").textContent = "查询失败，请刷新重试";
        if (!poll || pollFailures === 1) U.notify(U.message(error), true);
      }
    } finally {
      if (current === generation) { loading = false; $("task-list").setAttribute("aria-busy", "false"); updatePagination(); }
      schedulePoll();
    }
  }
  async function mutateTask(operation, snapshot) {
    await U.runExclusive(async () => {
      U.requireResolved();
      const id = taskId(snapshot);
      // 不信任旧列表按钮；在发起写操作之前重新核对来源和能力。
      const result = await U.query("tasks", "detail", { task_id: id });
      if (!U.plainObject(result.task) || taskId(result.task) !== id || !canMutate(result.task, operation)) throw new Error("任务来源或状态已经改变，此任务目前不可执行该操作。请刷新列表。");
      const params = { task_id: id };
      const request = U.requestId();
      let token;
      if (operation === "retry") {
        token = await U.confirmAction("retry", params, request, { title: "二次确认：重试任务与模型费用", description: "重试可能再次调用视觉模型或图像生成模型并计费。原任务已经发生的费用不会退回；本页面不会自动重试。", approve: "确认费用并重试" });
        if (!token) return;
      } else {
        const accepted = await U.ask({ title: "取消网页任务", description: "只会取消允许取消的后续处理。模型请求如果已经开始，费用不会退回；不能保证中止已经在执行的模型调用。", summary: `任务号：${id}`, approve: "确认取消任务", danger: true });
        if (!accepted) return;
      }
      const mutation = await U.action(operation, params, request, token);
      U.notify(operation === "cancel" ? "已提交取消请求，请以刷新后的任务状态为准。" : `重试请求已受理${mutation.task_id ? `，任务号：${mutation.task_id}` : ""}。`);
      await refreshTasks(false);
    }, operation === "cancel" ? "正在确认取消任务…" : "正在确认重试费用…");
    schedulePoll();
  }
  function applyFilters() {
    if (U.busy || loading) return;
    filters = { status: $("task-status-filter").value, origin: $("task-origin-filter").value, offset: 0, limit: Number($("task-page-size").value) };
    void refreshTasks(false);
  }
  function bind() {
    $("nav-gallery").addEventListener("click", () => U.navigate("gallery", { gallery_filters: galleryFilters }));
    $("apply-task-filters").addEventListener("click", applyFilters);
    $("refresh-tasks").addEventListener("click", () => { if (!U.busy) void refreshTasks(false); });
    $("previous-task-page").addEventListener("click", () => { if (U.busy || loading) return; filters.offset = Math.max(0, filters.offset - filters.limit); void refreshTasks(false); });
    $("next-task-page").addEventListener("click", () => { if (U.busy || loading) return; filters.offset += filters.limit; void refreshTasks(false); });
    $("task-poll").addEventListener("change", schedulePoll);
    $("task-close").addEventListener("click", closeTask);
    $("task-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeTask(); });
    $("task-refresh-detail").addEventListener("click", () => { if (task && !U.busy) readDetail(taskId(task)).catch((error) => U.notify(U.message(error), true)); });
    $("task-view-asset").addEventListener("click", () => { if (task?.asset_id) U.navigate("gallery", { asset_id: task.asset_id, gallery_filters: galleryFilters }); });
    $("task-cancel").addEventListener("click", () => task && mutateTask("cancel", task));
    $("task-retry").addEventListener("click", () => task && mutateTask("retry", task));
    document.addEventListener("photo:lock", () => { updatePagination(); updateDetailControls(); schedulePoll(); });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") schedulePoll();
      else if ($("task-poll").checked && !U.busy && !loading) void refreshTasks(true);
      else schedulePoll();
    });
    document.addEventListener("photo:receipt", () => { if (!U.busy && !loading && document.visibilityState !== "hidden") void refreshTasks(false); });
    window.addEventListener("pagehide", () => { disposed = true; if (pollingTimer !== null) clearTimeout(pollingTimer); });
  }
  async function start() {
    const state = await U.init();
    if (U.plainObject(state.gallery_filters)) galleryFilters = state.gallery_filters;
    bind();
    $("busy-label").textContent = "";
    updateDetailControls();
    await refreshTasks(false);
    if (typeof state.task_id === "string" && state.task_id) await openTask(state.task_id);
  }
  start().catch(U.fatal);
})();
