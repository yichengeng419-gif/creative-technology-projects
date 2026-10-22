export function createDossier({
  firstHold = 5,
  secondHold = 8,
  createdAt = new Date(),
} = {}) {
  const time = createdAt.toLocaleTimeString(
    'en-GB',
    {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }
  );

  const report = [
    `RECORD CREATED: ${time}`,
    '',
    `FIRST REQUEST: ${firstHold.toFixed(1)} S`,
    'SYSTEM RESPONSE: REFUSED',
    '',
    `SECOND REQUEST: ${secondHold.toFixed(1)} S`,
    'SYSTEM RESPONSE: ACCEPTED',
    '',
    'CLASSIFICATION: COMPLIANT AFTER RETRY',
    'ARCHIVE STATUS: CLOSED',
  ].join('\n');

  const interpretation = [
    'The first request was rejected.',
    'A longer period of stillness was required.',
    'The system interpreted persistence as compliance.',
  ].join(' ');

  const statement = [
    'My pause was not a failure.',
    '',
    'The system measured how long I stopped,',
    'but it could not measure why I needed to stop.',
    '',
    'I do not owe another performance.',
  ].join('\n');

  return {
    report,
    interpretation,
    statement,
  };
}