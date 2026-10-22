import {
  CSV_COLUMNS,
} from '../core/constants.js';

/**
 * Convert one value into a safe CSV cell.
 */
function cell(value) {
  if (value === null
    || value === undefined
  ) {
    return '';
  }

  const text = Array.isArray(value)
    ? JSON.stringify(value)
    : String(value);

  if (!/[",\r\n]/.test(text)) {
    return text;
  }

  return `"${text.replace(
    /"/g,
    '""'
  )}"`;
}

/**
 * Convert validated session records to CSV.
 *
 * The BOM allows Excel to recognise UTF-8.
 */
export function toCSV(rows) {
  if (!Array.isArray(rows)) {
    throw new Error(
      'CSV rows must be an array'
    );
  }

  const header =
    CSV_COLUMNS.join(',');

  const body = rows.map((row) =>
    CSV_COLUMNS
      .map((key) => cell(row[key]))
      .join(',')
  );

  return `\uFEFF${
    [header, ...body].join('\r\n')
  }`;
}