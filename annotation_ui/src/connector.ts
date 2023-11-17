import { DEVMODE } from './globals'

let SERVER_DATA_ROOT = DEVMODE ? "http://127.0.0.1:9000/queues/" : "queues/"
let SERVER_LOG_ROOT = DEVMODE ? "http://127.0.0.1:5000/" : "https://zouharvi.pythonanywhere.com/"

export async function load_data(): Promise<any> {
    let random_v = `?v=${Math.random()}`;
    let result :string = await $.ajax(
        SERVER_DATA_ROOT + globalThis.uid + ".jsonl" + random_v,
        {
            type: 'GET',
            contentType: 'application/text',
        }
    )
    result = result.trimEnd()
    result = "[" + result.replaceAll("\n", ",") + "]"
    result = JSON.parse(result)
    return result
}

export async function log_data(data): Promise<any> {
    if (globalThis.prolific_pid != undefined) {
        data["prolific_pid"] = globalThis.prolific_pid
    }

    let result = await $.ajax(
        SERVER_LOG_ROOT + "log",
        {
            data: JSON.stringify({
                project: "reading-comprehension-help",
                uid: globalThis.uid,
                prolific_pid: globalThis.prolific_pid,
                payload: JSON.stringify(data),
            }),
            type: 'POST',
            contentType: 'application/json',
        }
    )
    return result
}

export async function get_json(name: string): Promise<Array<object>> {
    let result = await $.ajax(
        name,
        {
            type: 'GET',
            contentType: 'application/text',
        }
    )
    result = result.trimEnd()
    result = "[" + result.replaceAll("\n", ",") + "]"
    result = JSON.parse(result)
    return result
}

export async function get_html(name: string): Promise<string> {
    return await $.ajax(
        name,
        {
            type: 'GET',
            contentType: 'text/html',
        }
    )
}